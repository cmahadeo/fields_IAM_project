# Case mapping and feature engineering ideas

## Use Case Mapping

| **Use Case**                        | **Data Needed**                                | **Feature Example**                                     |
| ----------------------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| Maker-Checker Violation             | Entitlements (type), Assignments, Applications | User has both `Submit_Payment` and `Approve_Payment`    |
| Admin + Operational Overlap         | Entitlements (type), Assignments               | User has `Root_Access` and `Process_Transactions`       |
| Unjustified Cross-Dept Access       | Departments, Employees, Assignments            | User in HR has access to Finance apps                   |
| High-Risk Apps Access Concentration | Applications (criticality), Assignments        | User with 5+ entitlements to high-risk apps             |
| Orphan Entitlements                 | Employees, Assignments                         | Assigned entitlement but no longer in role (job change) |


## Feature Engineering Ideas

| **Feature Name**               | **Description**                                                             |
| ------------------------------ | --------------------------------------------------------------------------- |
| `num_entitlements`             | Total number of entitlements per user                                       |
| `num_high_risk_entitlements`   | Count of entitlements with `risk_level` = "High"                            |
| `num_applications_accessed`    | Unique apps accessed by employee                                            |
| `access_duration_days`         | Days since `grant_date`                                                     |
| `cross_department_access_flag` | True if employee has entitlements across multiple departments' applications |
| `maker_checker_flag`           | True if user has both maker and checker entitlements on same app            |
| `admin_operational_overlap`    | True if user has both "Admin" and "Operational" types in entitlements       |
