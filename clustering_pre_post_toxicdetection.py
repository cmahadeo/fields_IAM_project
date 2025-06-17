import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def load_toxic_combinations_from_single_file(file_path="department_anomaly_results/all_departments_combined.csv"):
    """
    Load toxic combinations from single file containing all departments.

    Args:
        file_path: Path to the all_departments_combined.csv file

    TOXIC = is_anomaly=True AND same_application=True

    Returns:
        Dict of {department: {(ent1, ent2): entitlement_to_remove}}
    """
    print(f"🔍 Loading toxic combinations from single file: '{file_path}'...")

    if not os.path.exists(file_path):
        print(f"❌ File '{file_path}' not found")
        return {}

    try:
        # Load the combined file
        df = pd.read_csv(file_path)

        print(f"📊 Loaded {len(df)} combinations from {os.path.basename(file_path)}")
        print(f"📋 Columns: {df.columns.tolist()}")

        # Check required columns
        required_cols = ['is_anomaly', 'same_application', 'entitlement_1', 'entitlement_2']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return {}

        # Check if there's a department column to segment by
        dept_col = None
        possible_dept_cols = ['department', 'name_department', 'dept', 'batch_id']
        for col in possible_dept_cols:
            if col in df.columns:
                dept_col = col
                break

        if not dept_col:
            print(f"⚠️  No department column found. Available columns: {df.columns.tolist()}")
            print(f"ℹ️  Will treat all data as one department: 'All_Departments'")
            # Treat everything as one department
            departments = {'All_Departments': df}
        else:
            print(f"✅ Using '{dept_col}' as department column")
            # Group by department
            departments = {dept: group for dept, group in df.groupby(dept_col)}

        # Define TOXIC for all departments
        toxic_combinations = {}

        for dept_name, dept_df in departments.items():
            print(f"  🔄 Processing {dept_name}...")

            # Define TOXIC: anomalous AND same application
            toxic_mask = (dept_df['is_anomaly'] == True) & (dept_df['same_application'] == True)
            toxic_df = dept_df[toxic_mask]

            print(f"    🚨 Found {len(toxic_df)} TOXIC combinations (anomalous + same app)")

            if len(toxic_df) == 0:
                print(f"    ℹ️  No toxic combinations for {dept_name}")
                continue

            # Process toxic combinations to determine what to remove
            dept_toxic_removals = {}

            for _, row in toxic_df.iterrows():
                ent1 = row['entitlement_1']
                ent2 = row['entitlement_2']

                # Determine privilege levels
                ent1_level = get_privilege_level(ent1)
                ent2_level = get_privilege_level(ent2)

                # Keep the least privileged, remove the more privileged
                if ent1_level < ent2_level:  # ent1 is less privileged
                    entitlement_to_remove = ent2
                elif ent2_level < ent1_level:  # ent2 is less privileged
                    entitlement_to_remove = ent1
                else:  # Same privilege level, remove one arbitrarily
                    entitlement_to_remove = ent2  # Remove second one

                # Store the entitlement to remove for this combination
                combo_key = tuple(sorted([ent1, ent2]))
                dept_toxic_removals[combo_key] = entitlement_to_remove

            toxic_combinations[dept_name] = dept_toxic_removals

            print(f"    ✅ {dept_name}: {len(dept_toxic_removals)} toxic entitlement pairs to clean")

        print(f"🎉 Loaded toxic combinations for {len(toxic_combinations)} departments")
        return toxic_combinations

    except Exception as e:
        print(f"❌ Error loading {file_path}: {str(e)}")
        return {}

def get_privilege_level(entitlement_name):
    """
    Get privilege level for an entitlement.
    Returns: 1=read, 2=write, 3=execute, 4=admin
    """
    ent_lower = entitlement_name.lower()

    if 'admin' in ent_lower:
        return 4
    elif 'execute' in ent_lower:
        return 3
    elif 'write' in ent_lower:
        return 2
    elif 'read' in ent_lower:
        return 1
    else:
        return 1  # Default to read

def binarize_entitlement_data_by_department(df):
    """
    Binarize the main entitlement dataframe by department.

    Args:
        df: DataFrame with columns ['employee_id', 'entitlement_id', 'name_department']

    Returns:
        Dict of {department: binarized_dataframe}
    """
    print(f"🔄 Binarizing entitlement data by department...")

    # Validate columns
    required_cols = ['employee_id', 'entitlement_id', 'name_department']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return {}

    departments = df['name_department'].unique()
    binarized_data = {}

    print(f"📊 Processing {len(departments)} departments...")

    for dept in departments:
        print(f"  🔄 Binarizing {dept}...")

        # Filter data for department
        dept_df = df[df['name_department'] == dept].copy()

        if len(dept_df) == 0:
            print(f"    ⚠️  No data for {dept}")
            continue

        try:
            # Remove duplicates first
            dept_df = dept_df[['employee_id', 'entitlement_id']].drop_duplicates()

            # Create binary matrix using crosstab
            binary_matrix = pd.crosstab(
                dept_df['employee_id'],
                dept_df['entitlement_id']
            )

            # Convert to binary (1 where relationship exists, 0 otherwise)
            binary_matrix = (binary_matrix > 0).astype(int)

            # Reset index to make employee_id a column
            binary_matrix = binary_matrix.reset_index()

            binarized_data[dept] = binary_matrix

            print(f"    ✅ {dept}: {binary_matrix.shape[0]} employees × {binary_matrix.shape[1]-1} entitlements")

        except Exception as e:
            print(f"    ❌ Error binarizing {dept}: {str(e)}")

    print(f"🎉 Binarized data for {len(binarized_data)} departments")
    return binarized_data

def clean_binarized_data_with_toxic_combinations(binarized_data, toxic_combinations):
    """
    Clean binarized data by removing toxic entitlements.
    For each toxic combination, remove the higher privilege entitlement from ALL employees who have both.

    Args:
        binarized_data: Dict of {department: binarized_dataframe}
        toxic_combinations: Dict of {department: {(ent1, ent2): entitlement_to_remove}}

    Returns:
        Dict of {department: cleaned_binarized_dataframe}
    """
    print(f"🧹 Cleaning binarized data using toxic combinations...")

    cleaned_data = {}
    total_removals_all_depts = 0

    for dept, binary_df in binarized_data.items():
        print(f"  🔄 Cleaning {dept}...")

        # Start with copy of original
        cleaned_df = binary_df.copy()

        # Get toxic combinations for this department
        dept_toxic = toxic_combinations.get(dept, {})

        if not dept_toxic:
            print(f"    ℹ️  No toxic combinations for {dept}")
            cleaned_data[dept] = cleaned_df
            continue

        # Apply removals
        removals_applied = 0

        for (ent1, ent2), entitlement_to_remove in dept_toxic.items():
            # Find employees who have BOTH entitlements (toxic combination)
            if ent1 in cleaned_df.columns and ent2 in cleaned_df.columns and entitlement_to_remove in cleaned_df.columns:
                # Employees with both entitlements
                both_mask = (cleaned_df[ent1] == 1) & (cleaned_df[ent2] == 1)
                employees_with_both = both_mask.sum()

                if employees_with_both > 0:
                    # Remove the higher privilege entitlement from these employees
                    cleaned_df.loc[both_mask, entitlement_to_remove] = 0
                    removals_applied += employees_with_both

                    print(f"    🗑️  Removed {entitlement_to_remove} from {employees_with_both} employees (toxic with {ent1 if entitlement_to_remove != ent1 else ent2})")

        cleaned_data[dept] = cleaned_df
        total_removals_all_depts += removals_applied

        print(f"    ✅ {dept}: Applied {removals_applied} entitlement removals")

    print(f"🎉 Cleaned data for {len(cleaned_data)} departments")
    print(f"📊 Total entitlements removed across all departments: {total_removals_all_depts}")
    return cleaned_data

def perform_clustering_analysis(data_dict, analysis_name="Analysis"):
    """
    Perform clustering analysis on binarized data.

    Args:
        data_dict: Dict of {department: dataframe}
        analysis_name: Name for this analysis (e.g., "Original" or "Cleaned")

    Returns:
        Dict of {department: clustering_results}
    """
    print(f"📊 Performing {analysis_name} clustering analysis...")

    results = {}

    for dept, dept_df in data_dict.items():
        print(f"  🔄 Analyzing {dept}...")

        if len(dept_df) == 0:
            print(f"    ⚠️  No data for {dept}")
            continue

        # Prepare features (exclude employee_id)
        feature_cols = [col for col in dept_df.columns if col != 'employee_id']
        X = dept_df[feature_cols].values

        # Skip if not enough samples or features
        if len(X) < 2:
            print(f"    ⚠️  {dept}: Less than 2 employees, skipping")
            continue

        if X.shape[1] < 2:
            print(f"    ⚠️  {dept}: Less than 2 entitlements, skipping")
            continue

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Test different cluster numbers (2 to min(17, n_samples-1))
        max_k = min(17, len(X) - 1)
        k_range = range(2, max_k + 1)

        inertias = []
        silhouette_scores = []

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                cluster_labels = kmeans.fit_predict(X_scaled)

                inertias.append(kmeans.inertia_)

                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:
                    sil_score = silhouette_score(X_scaled, cluster_labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)

            except Exception as e:
                print(f"    ⚠️  Error with k={k}: {str(e)}")
                break

        # Find best silhouette score
        if silhouette_scores:
            best_k_idx = np.argmax(silhouette_scores)
            best_k = list(k_range)[best_k_idx]
            best_silhouette = silhouette_scores[best_k_idx]
        else:
            best_k = 2
            best_silhouette = 0

        results[dept] = {
            'k_range': list(k_range)[:len(inertias)],
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'best_k': best_k,
            'best_silhouette': best_silhouette,
            'n_samples': len(X),
            'n_features': len(feature_cols)
        }

        print(f"    ✅ {dept}: Best k={best_k}, Silhouette={best_silhouette:.4f}")

    print(f"🎉 {analysis_name} analysis complete for {len(results)} departments")
    return results

def create_detailed_comparison_summary(original_results, cleaned_results, toxic_combinations, binarized_data, cleaned_data):
    """
    Create detailed summary comparison of before vs after cleaning (OPTIMIZED VERSION).

    Returns:
        pandas DataFrame with comparison metrics
    """
    print(f"📋 Creating detailed comparison summary...")

    summary_data = []

    for dept in original_results.keys():
        if dept in cleaned_results:
            orig = original_results[dept]
            clean = cleaned_results[dept]

            # Calculate improvements
            silhouette_improvement = clean['best_silhouette'] - orig['best_silhouette']
            k_change = clean['best_k'] - orig['best_k']

            # Get toxic combination stats (fast lookup)
            dept_toxic = toxic_combinations.get(dept, {})
            toxic_pairs = len(dept_toxic)

            # OPTIMIZED: Calculate actual entitlements removed using vectorized operations
            entitlements_removed = 0
            employees_affected = 0

            if dept in binarized_data and dept in cleaned_data:
                orig_df = binarized_data[dept]
                clean_df = cleaned_data[dept]

                # Get feature columns (exclude employee_id)
                feature_cols = [col for col in orig_df.columns if col != 'employee_id' and col in clean_df.columns]

                if feature_cols:
                    # FAST: Vectorized comparison of all entitlements at once
                    orig_values = orig_df[feature_cols].values
                    clean_values = clean_df[feature_cols].values

                    # Count where 1s became 0s (removals)
                    removals_matrix = (orig_values == 1) & (clean_values == 0)
                    entitlements_removed = int(removals_matrix.sum())

                    # Count employees with any changes (much faster)
                    employees_with_changes = (removals_matrix.sum(axis=1) > 0).sum()
                    employees_affected = int(employees_with_changes)

            # Calculate improvement percentage
            improvement_percentage = 0
            if orig['best_silhouette'] > 0:
                improvement_percentage = (silhouette_improvement / orig['best_silhouette']) * 100

            summary_data.append({
                'Department': dept,
                'Employees': orig['n_samples'],
                'Entitlements': orig['n_features'],
                'Original_Best_K': orig['best_k'],
                'Cleaned_Best_K': clean['best_k'],
                'K_Change': f"{k_change:+d}" if k_change != 0 else "0",
                'Original_Silhouette': f"{orig['best_silhouette']:.4f}",
                'Cleaned_Silhouette': f"{clean['best_silhouette']:.4f}",
                'Silhouette_Improvement': f"{silhouette_improvement:+.4f}",
                'Improvement_%': f"{improvement_percentage:+.1f}%",
                'Toxic_Combination_Pairs': toxic_pairs,
                'Employees_Affected': employees_affected,
                'Entitlements_Removed': entitlements_removed,
                'Toxic_Rate_%': f"{(employees_affected/orig['n_samples']*100):.1f}%" if orig['n_samples'] > 0 else "0%"
            })

            print(f"    ✅ {dept}: {entitlements_removed} removals, {employees_affected} employees affected")

    summary_df = pd.DataFrame(summary_data)
    print(f"✅ Summary created for {len(summary_df)} departments")

    return summary_df

def complete_entitlement_cleaning_pipeline(data, file_path=None):
    """
    Complete pipeline: Load toxic combinations from single file, binarize data, clean, and analyze.

    Args:
        data: Main entitlement DataFrame with ['employee_id', 'entitlement_id', 'name_department']
        file_path: Path to the all_departments_combined.csv file (if None, uses default path)

    Returns:
        Tuple of (original_results, cleaned_results, summary_df, binarized_data, cleaned_data, toxic_combinations)
    """
    print("🚀 Starting complete entitlement cleaning and clustering pipeline...")
    print("=" * 70)

    # Default file path if not provided
    if file_path is None:
        file_path = "department_anomaly_results/all_departments_combined.csv"

    # Check if the path is a directory (common mistake)
    if os.path.isdir(file_path):
        # If it's a directory, look for the CSV file inside it
        if file_path.endswith('/'):
            file_path = file_path + "all_departments_combined.csv"
        else:
            file_path = file_path + "/all_departments_combined.csv"

        print(f"📁 Detected directory path, using: {file_path}")

    # Step 1: Load toxic combinations from single file
    print(f"\n1️⃣  STEP 1: Loading toxic combinations from: {file_path}")
    toxic_combinations = load_toxic_combinations_from_single_file(file_path)

    if not toxic_combinations:
        print("❌ No toxic combinations found")
        return None, None, None, None, None, None

    # Step 2: Binarize main entitlement data by department
    print("\n2️⃣  STEP 2: Binarizing main entitlement data by department")
    binarized_data = binarize_entitlement_data_by_department(data)

    if not binarized_data:
        print("❌ Failed to binarize data")
        return None, None, None, None, None, None

    # Step 3: Perform original clustering
    print("\n3️⃣  STEP 3: Original clustering analysis (before cleaning)")
    original_results = perform_clustering_analysis(binarized_data, "Original")

    # Step 4: Clean data using toxic combinations
    print("\n4️⃣  STEP 4: Cleaning data using toxic combinations")
    cleaned_data = clean_binarized_data_with_toxic_combinations(binarized_data, toxic_combinations)

    # Step 5: Perform cleaned clustering
    print("\n5️⃣  STEP 5: Cleaned clustering analysis (after cleaning)")
    cleaned_results = perform_clustering_analysis(cleaned_data, "Cleaned")

    # Step 6: Create detailed comparison summary
    print("\n6️⃣  STEP 6: Creating detailed comparison summary")
    summary_df = create_detailed_comparison_summary(original_results, cleaned_results, toxic_combinations, binarized_data, cleaned_data)

    print("\n🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"📊 DETAILED SUMMARY RESULTS:")
    print(summary_df.to_string(index=False))

    # Show key insights
    if len(summary_df) > 0:
        total_entitlements_removed = summary_df['Entitlements_Removed'].astype(int).sum()
        total_employees_affected = summary_df['Employees_Affected'].astype(int).sum()

        print(f"\n🎯 KEY INSIGHTS:")
        print(f"   📊 Total entitlements removed: {total_entitlements_removed:,}")
        print(f"   👥 Total employees affected: {total_employees_affected:,}")

        # Find departments with improvements
        summary_df['Improvement_Numeric'] = summary_df['Improvement_%'].str.replace('%', '').astype(float)
        improvements = summary_df[summary_df['Improvement_Numeric'] > 0]
        degradations = summary_df[summary_df['Improvement_Numeric'] < 0]

        if len(improvements) > 0:
            print(f"   ✅ {len(improvements)} departments improved")
            best_improvement = improvements.loc[improvements['Improvement_Numeric'].idxmax()]
            print(f"   🏆 Best improvement: {best_improvement['Department']} ({best_improvement['Improvement_%']})")

        if len(degradations) > 0:
            print(f"   ⚠️  {len(degradations)} departments degraded")
            worst_degradation = degradations.loc[degradations['Improvement_Numeric'].idxmin()]
            print(f"   📉 Most degradation: {worst_degradation['Department']} ({worst_degradation['Improvement_%']})")

    return original_results, cleaned_results, summary_df, binarized_data, cleaned_data, toxic_combinations

