import pandas as pd
import numpy as np

def generateDatasetWithClass(destination, source):
  file = open(destination, "w")
  df = pd.read_csv(source, header=None, delimiter=' ')
  df = df.iloc[:,:-1]
  print(df)

  for i in range(1, 401, 1):
    class_of_instance = 1

    if i <= 50:
      class_of_instance = 1
    elif i <= 100:
      class_of_instance = 0
    elif i <= 150:
      class_of_instance = 1
    elif i <= 200:
      class_of_instance = 0
    elif i <= 250:
      class_of_instance = 1
    elif i <= 300:
      class_of_instance = 0
    elif i <= 350:
      class_of_instance = 1
    else:
      class_of_instance = 0

    df_with_class = pd.DataFrame(np.append(df.iloc[i-1], [class_of_instance])).transpose()
    np.savetxt(file, df_with_class, delimiter=',')

  file.close()