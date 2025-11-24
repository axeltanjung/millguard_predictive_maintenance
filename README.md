# MillGuard - Predictive Maintenance
## Dataset Features

The dataset contains simulated production and condition-monitoring variables used for predictive maintenance modeling.

| Feature                     | Description                                                                                                                                                                                               |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **UID**                     | Unique identifier ranging from 1 to 10,000.                                                                                                                                                               |
| **Product ID**              | A composite identifier consisting of a quality code (`L`, `M`, or `H`) — representing low (50% of samples), medium (30%), and high (20%) quality variants — followed by a variant-specific serial number. |
| **Type**                    | The product quality type extracted from the Product ID (`L`, `M`, or `H`).                                                                                                                                |
| **Air Temperature [K]**     | Simulated using a random walk process, normalized to a standard deviation of 2 K around a mean of 300 K.                                                                                                  |
| **Process Temperature [K]** | Generated from a random walk process normalized to 1 K standard deviation, added to the air temperature plus 10 K.                                                                                        |
| **Rotational Speed [rpm]**  | Calculated based on a power of 2860 W with additional Gaussian noise.                                                                                                                                     |
| **Torque [Nm]**             | Normally distributed around 40 Nm with a standard deviation of 10 Nm, ensuring no negative values.                                                                                                        |
| **Tool Wear [min]**         | Tool wear increases based on product quality: +2 min for L, +3 min for M, and +5 min for H variants.                                                                                                      |
| **Machine Failure**         | Binary label indicating whether the machine failed during this process (1 = failure, 0 = no failure). Failures arise from one or more of the independent failure modes below.                             |
