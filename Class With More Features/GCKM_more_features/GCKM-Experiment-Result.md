# Experiment Results for GaussianCopulaKmeansSynthesizer

## Introduction
This document presents the results of the experiments conducted with the `GaussianCopulaKmeansSynthesizer`. The aim was to evaluate the performance of the synthesizer across various datasets, emphasizing the efficiency of its unique K-means initialization approach.

## Objectives
- To assess the accuracy and efficiency of the `GaussianCopulaKmeansSynthesizer` in generating synthetic data.
- To compare the performance of the synthesizer with the Synthetic Data Vault (SDV) model.
- To demonstrate the effectiveness of the quantile-based approach for K-means initialization.

## Methodology
### Datasets Used
- American Income
  | age | workclass       | fnlwgt | education  | education-num | marital-status       | occupation         | relationship    | race  | sex    | capital-gain | capital-loss | hours-per-week | native-country | salary |
  |-----|-----------------|--------|------------|---------------|----------------------|--------------------|-----------------|-------|--------|--------------|--------------|----------------|----------------|--------|
  | 39  | State-gov       | 77516  | Bachelors  | 13            | Never-married        | Adm-clerical       | Not-in-family   | White | Male   | 2174         | 0            | 40             | United-States  | <=50K  |
  | 50  | Self-emp-not-inc| 83311  | Bachelors  | 13            | Married-civ-spouse   | Exec-managerial    | Husband         | White | Male   | 0            | 0            | 13             | United-States  | <=50K  |
  | 38  | Private         | 215646 | HS-grad    | 9             | Divorced             | Handlers-cleaners  | Not-in-family   | White | Male   | 0            | 0            | 40             | United-States  | <=50K  |
  | 53  | Private         | 234721 | 11th       | 7             | Married-civ-spouse   | Handlers-cleaners  | Husband         | Black | Male   | 0            | 0            | 40             | United-States  | <=50K  |
  | 28  | Private         | 338409 | Bachelors  | 13            | Married-civ-spouse   | Prof-specialty     | Wife            | Black | Female | 0            | 0            | 40             | Cuba           | <=50K  |
- Travel Insurance
  | Agency | Agency Type   | Distribution Channel | Product Name                    | Claim | Duration | Destination | Net Sales | Commision (in value) | Gender | Age |
  |--------|---------------|----------------------|---------------------------------|-------|----------|-------------|-----------|-----------------------|--------|-----|
  | CBH    | Travel Agency | Offline              | Comprehensive Plan              | No    | 186      | MALAYSIA    | -29.0     | 9.57                  | F      | 81  |
  | CBH    | Travel Agency | Offline              | Comprehensive Plan              | No    | 186      | MALAYSIA    | -29.0     | 9.57                  | F      | 71  |
  | CWT    | Travel Agency | Online               | Rental Vehicle Excess Insurance | No    | 65       | AUSTRALIA   | -49.5     | 29.70                 | NaN    | 32  |
  | CWT    | Travel Agency | Online               | Rental Vehicle Excess Insurance | No    | 60       | AUSTRALIA   | -39.6     | 23.76                 | NaN    | 32  |
  | CWT    | Travel Agency | Online               | Rental Vehicle Excess Insurance | No    | 79       | ITALY       | -19.8     | 11.88                 | NaN    | 41  |  
- Fraud Insurance
  | months_as_customer | age | policy_number | policy_bind_date     | policy_state | policy_csl | policy_deductable | policy_annual_premium | umbrella_limit | insured_zip | ... | witnesses | police_report_available | total_claim_amount | injury_claim | property_claim | vehicle_claim | auto_make  | auto_model | auto_year | fraud_reported |
  |--------------------|-----|---------------|----------------------|--------------|------------|-------------------|-----------------------|----------------|-------------|-----|-----------|-------------------------|--------------------|--------------|---------------|--------------|------------|------------|-----------|---------------|
  | 328                | 48  | 521585        | 2014-10-17 00:00:00  | OH           | 250/500    | 1000              | 1406.91               | 0              | 466132      | ... | 2         | YES                     | 71610              | 6510         | 13020         | 52080         | Saab       | 92x        | 2004      | Y             |
  | 228                | 42  | 342868        | 2006-06-27 00:00:00  | IN           | 250/500    | 2000              | 1197.22               | 5000000        | 468176      | ... | 0         | ?                       | 5070               | 780          | 780           | 3510          | Mercedes   | E400       | 2007      | Y             |
  | 134                | 29  | 687698        | 2000-09-06 00:00:00  | OH           | 100/300    | 2000              | 1413.14               | 5000000        | 430632      | ... | 3         | NO                      | 34650              | 7700         | 3850          | 23100         | Dodge      | RAM        | 2007      | N             |
  | 256                | 41  | 227811        | 1990-05-25 00:00:00  | IL           | 250/500    | 2000              | 1415.74               | 6000000        | 608117      | ... | 2         | NO                      | 63400              | 6340         | 6340          | 50720         | Chevrolet  | Tahoe      | 2014      | Y             |
  | 228                | 44  | 367455        | 2014-06-06 00:00:00  | IL           | 500/1000   | 1000              | 1583.91               | 6000000        | 610706      | ... | 1         | NO                      | 6500               | 1300         | 650           | 4550          | Accura     | RSX        | 2009      | N             |


Each dataset was subjected to a series of tests to evaluate the performance of the synthesizer.

### Experimental Setup
The GaussianCopulaKmeansSynthesizer was configured with specific parameters tailored to each dataset. The model's unique method of K-means initialization was a focal point in these experiments.

### Procedures
- **Data Preprocessing:** Load the data into the model, and perform the command to let the model do the data preprocessing.
- **Model Training:** The synthesizer was trained on each dataset.
- **Performance Evaluation:** The model's output was compared against the original data and the SDV model.

## Results
### American Income Dataset (32561 rows × 15 columns)
- **GaussianCopulaKmeansSynthesizer:**
  |     | age        | workclass         | fnlwgt       | education    | education-num | marital-status      | occupation        | relationship   | race  | sex    | capital-gain | capital-loss | hours-per-week | native-country | salary |
  |-----|------------|-------------------|--------------|--------------|---------------|---------------------|-------------------|----------------|-------|--------|--------------|--------------|----------------|----------------|--------|
  | 0   | 39.182060  | Private           | 226003.837575| 11th         | 9.919496      | Never-married       | Other-service     | Own-child      | White | Male   | 3694.539617  | 281.685453   | 1.000000       | United-States  | <=50K  |
  | 1   | 38.928003  | Private           | 309924.271490| Masters      | 9.686614      | Married-civ-spouse  | Exec-managerial   | Husband        | White | Male   | 1774.837063  | 671.379862   | 52.519362      | United-States  | <=50K  |
  | 2   | 26.368483  | ?                 | 12285.000000 | Some-college | 7.866581      | Married-civ-spouse  | Sales             | Husband        | White | Female | 322.474967   | 191.510807   | 42.273840      | United-States  | <=50K  |
  | 3   | 62.928026  | Private           | 280589.010978| Assoc-voc    | 7.901175      | Divorced            | Adm-clerical      | Not-in-family  | White | Male   | 3296.269710  | 299.566982   | 38.437363      | United-States  | <=50K  |
  | 4   | 38.609475  | Self-emp-not-inc  | 12285.000000 | HS-grad      | 10.982584     | Married-civ-spouse  | Exec-managerial   | Husband        | White | Male   | 4410.377013  | 408.454806   | 32.350629      | United-States  | <=50K  |

  ```
  Creating report: 100%|██████████| 4/4 [00:00<00:00,  4.66it/s]

  Overall Quality Score: 78.59%

  Properties:
  Column Shapes: 78.78%
  Column Pair Trends: 78.4%
  ```
- **SDV Model:**
  |     | age | workclass       | fnlwgt | education | education-num | marital-status    | occupation        | relationship | race | sex    | capital-gain | capital-loss | hours-per-week | native-country | salary |
  |-----|-----|-----------------|--------|-----------|---------------|-------------------|-------------------|--------------|------|--------|--------------|--------------|----------------|----------------|--------|
  | 0   | 52  | Self-emp-not-inc| 147899 | 11th      | 10            | Married-civ-spouse| Tech-support      | Husband      | White| Male   | 35253        | 190          | 66             | United-States  | <=50K  |
  | 1   | 22  | Self-emp-not-inc| 82206  | 11th      | 10            | Never-married     | Sales             | Wife         | Black| Male   | 57472        | 192          | 38             | United-States  | <=50K  |
  | 2   | 28  | Private         | 332938 | 9th       | 6             | Never-married     | Handlers-cleaners | Not-in-family| Black| Male   | 35           | 16           | 30             | Jamaica        | <=50K  |
  | 3   | 40  | Private         | 176269 | Bachelors | 9             | Never-married     | Prof-specialty    | Husband      | White| Female | 28849        | 0            | 46             | United-States  | <=50K  |
  | 4   | 63  | Local-gov       | 270964 | 9th       | 7             | Married-civ-spouse| Craft-repair      | Unmarried    | White| Male   | 85673        | 0            | 28             | Puerto-Rico    | <=50K  |

  ```
  Creating report: 100%|██████████| 4/4 [00:00<00:00,  4.30it/s]

  Overall Quality Score: 72.42%
  
  Properties:
  Column Shapes: 75.37%
  Column Pair Trends: 69.47%
  ```
- **Observations:**
  The score for column shapes has seen only marginal improvement, which is quite expected, as the fitting process relies on standard distributions.

### Travel Insurance Dataset (63326 rows × 11 columns)
- **GaussianCopulaKmeansSynthesizer:**
  |     | Agency | Agency Type    | Distribution Channel | Product Name             | Claim | Duration   | Destination | Net Sales | Commision (in value) | Gender | Age      |
  |-----|--------|----------------|----------------------|--------------------------|-------|------------|-------------|-----------|----------------------|--------|---------|
  | 0   | JZI    | Airlines       | Online               | Silver Plan              | No    | 106.096422 | SINGAPORE   | 84.233178 | 6.985795e+01         | F      | 46.392377|
  | 1   | CWT    | Travel Agency  | Online               | 1 way Comprehensive Plan | No    | 1.637761   | CHINA       | 42.693945 | 4.938111e+01         | F      | 30.531097|
  | 2   | CWT    | Travel Agency  | Online               | Bronze Plan              | No    | -2.000000  | CHINA       | -18.21259 | 9.626817e+00         | NaN    | 45.415680|
  | 3   | C2B    | Travel Agency  | Online               | Value Plan               | No    | 122.169075 | THAILAND    | 94.411799 | 6.211278e+01         | NaN    | 47.805159|
  | 4   | EPX    | Travel Agency  | Online               | Cancellation Plan        | No    | 20.324568  | HONG KONG   | 76.352906 | 1.089564e-07         | NaN    | 15.799867|

  ```
  Creating report: 100%|██████████| 4/4 [00:00<00:00,  4.42it/s]

  Overall Quality Score: 83.85%
  
  Properties:
  Column Shapes: 85.43%
  Column Pair Trends: 82.26%
  ```
- **SDV Model:**
  |     | Agency | Agency Type    | Distribution Channel | Product Name                   | Claim | Duration | Destination | Net Sales | Commision (in value) | Gender | Age |
  |-----|--------|----------------|----------------------|--------------------------------|-------|----------|-------------|-----------|----------------------|--------|-----|
  | 0   | EPX    | Travel Agency  | Offline              | Silver Plan                    | No    | 8        | JAPAN       | 21.61     | 5.29                 | NaN    | 28  |
  | 1   | JZI    | Airlines       | Offline              | Rental Vehicle Excess Insurance| No    | 149      | VIET NAM    | 92.36     | 79.83                | NaN    | 51  |
  | 2   | CWT    | Travel Agency  | Offline              | Comprehensive Plan             | No    | 57       | DENMARK     | 107.28    | 33.11                | NaN    | 37  |
  | 3   | RAB    | Airlines       | Offline              | Single Trip Travel Protect Silver | No | 148  | INDONESIA   | 20.63     | 12.76                | NaN    | 48  |
  | 4   | JZI    | Travel Agency  | Offline              | Bronze Plan                    | No    | 83       | VIET NAM    | 53.62     | 18.63                | NaN    | 38  |

  ```
  Creating report: 100%|██████████| 4/4 [00:00<00:00,  4.50it/s]

  Overall Quality Score: 60.21%
  
  Properties:
  Column Shapes: 67.94%
  Column Pair Trends: 52.48%
  ```
- **Observations:** The score for column-pair-trend increased alot as the expect

### Fraud Insurance Dataset (1000 rows × 39 columns)
- **GaussianCopulaKmeansSynthesizer:**
  | months_as_customer | age       | policy_number | policy_bind_date | policy_state | policy_csl | policy_deductable | policy_annual_premium | umbrella_limit  | insured_zip | ... | witnesses | police_report_available | total_claim_amount | injury_claim  | property_claim | vehicle_claim | auto_make | auto_model | auto_year   | fraud_reported |
  |--------------------|-----------|---------------|------------------|--------------|------------|-------------------|-----------------------|-----------------|-------------|-----|-----------|-------------------------|--------------------|---------------|----------------|--------------|-----------|------------|-------------|---------------|
  | 410.318409         | 37.306267 | 766193        | 2012-06-27       | IN           | 100/300    | 683.177255        | 1198.576205           | 1.677383e-05    | 430104.0    | ... | 3.000000  | NO                      | 56674.500063       | 8143.038326   | 6178.960537   | 42358.661354 | Suburu    | Neon       | 1999.159802 | N             |
  | 300.830798         | 48.156783 | 132902        | 1992-10-19       | IN           | 100/300    | 1999.999623       | 1303.563734           | 1.000000e+07    | 430104.0    | ... | 0.180559  | NO                      | 51142.321764       | 10316.129364  | 3473.699062   | 37367.980449 | Suburu    | A3         | 1996.504191 | N             |
  | 260.513778         | 36.447316 | 379268        | 2003-04-22       | IN           | 250/500    | 1882.032696       | 1153.168828           | 7.189771e-06    | 430104.0    | ... | 0.000106  | NO                      | 52004.238955       | 10086.836213  | 9141.805248   | 32784.753461 | Honda     | Passat     | 1996.257238 | N             |
  | 191.310823         | 56.771964 | 491170        | 2009-11-11       | IN           | 250/500    | 1863.714231       | 1282.652137           | 1.124061e-05    | 430104.0    | ... | 2.954448  | ?                       | 49754.458427       | 8098.012455   | 6545.580617   | 35072.267617 | Saab      | A3         | 2002.905445 | N             |
  | 315.249733         | 32.922776 | 110084        | 2007-10-18       | IL           | 250/500    | 500.000000        | 1570.645903           | 5.655428e-09    | 430104.0    | ... | 2.305309  | NO                      | 54666.363904       | 8711.199704   | 7547.313295   | 38419.724137 | Saab      | X5         | 2014.433132 | N             |

  ```
  Creating report: 100%|██████████| 4/4 [00:13<00:00,  3.40s/it]

  Overall Quality Score: 72.17%
  
  Properties:
  Column Shapes: 78.73%
  Column Pair Trends: 65.6%
  ```
- **SDV Model:**
  | months_as_customer | age | policy_number | policy_bind_date     | policy_state | policy_csl | policy_deductable | policy_annual_premium | umbrella_limit | insured_zip | ... | witnesses | police_report_available | total_claim_amount | injury_claim | property_claim | vehicle_claim | auto_make | auto_model | auto_year | fraud_reported |
  |--------------------|-----|---------------|----------------------|--------------|------------|-------------------|-----------------------|----------------|-------------|-----|-----------|-------------------------|--------------------|--------------|---------------|--------------|-----------|------------|-----------|---------------|
  | 312                | 49  | 627167        | 2005-12-10 00:00:00  | IL           | 500/1000   | 1929              | 1115.94               | 9998427        | 585665      | ... | 1         | YES                     | 63360              | 17053        | 5935          | 44109         | BMW       | Corolla    | 2001      | N             |
  | 452                | 53  | 184494        | 2003-01-23 00:00:00  | OH           | 100/300    | 1962              | 1181.32               | -991599        | 448719      | ... | 0         | ?                       | 61012              | 8590         | 5161          | 46159         | Saab      | Camry      | 2004      | N             |
  | 413                | 54  | 665397        | 2003-09-13 00:00:00  | IN           | 100/300    | 1873              | 1357.32               | -818428        | 430104      | ... | 3         | NO                      | 43949              | 3969         | 22048         | 25404         | Jeep      | Malibu     | 2007      | Y             |
  | 406                | 59  | 344792        | 1992-04-04 00:00:00  | IL           | 250/500    | 536               | 671.14                | 2084764        | 430104      | ... | 0         | YES                     | 5573               | 590          | 9             | 4051          | Suburu    | Escape     | 1996      | Y             |
  | 2                  | 23  | 714798        | 1994-06-21 00:00:00  | IN           | 500/1000   | 1999              | 1525.45               | 7540826        | 430104      | ... | 1         | YES                     | 27524              | 7105         | 1172          | 19339         | Honda     | Jetta      | 2002      | Y             |

  ```
  Creating report: 100%|██████████| 4/4 [00:08<00:00,  2.21s/it]

  Overall Quality Score: 78.2%
  
  Properties:
  Column Shapes: 84.3%
  Column Pair Trends: 72.09%
  ```
- **Observations:** In the case of this particular dataset, the performance of the GaussianCopulaKmeansSynthesizer was not optimal, potentially attributable to the limited size of the dataset.

## Discussion
- The `GaussianCopulaKmeansSynthesizer` showed [summary of findings, e.g., improved accuracy, faster processing times].
- The quantile-based approach for K-means initialization provided [benefits observed].
- Comparison with the SDV model revealed improvement on the column-pair trends.

## Conclusion
The experimental results demonstrate the `GaussianCopulaKmeansSynthesizer`'s effectiveness in synthesizing data across various insurance datasets. The detailed evaluation including plots and self-defined metrics could be found at:
- American Income Evaluation: [American_Income_eval.ipynb](GCKM_America_income_eval.ipynb)
- Travel_Insurance Evaluation: [travel_insurance_eval.ipynb](GCKM_travel_ins_eval.ipynb)
- Fraud Insurance Evaluation: [fraud_insurance_claim_evaluation.ipynb](GCKM_fraud_insurance_evaluation.ipynb)

## Future Work
- Explore the application of the synthesizer to more diverse datasets.
- Further optimize the K-means initialization process for larger datasets.

