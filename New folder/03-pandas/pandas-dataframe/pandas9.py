import pandas as pd
import numpy as np
print("Example csv file : \n")
print(pd.read_csv('example.csv'))

df=pd.read_csv('example.csv')
print("Datafram from example.csv : \n",df)

df.to_csv('Output.csv')
df.to_csv('Output_false_index.csv',index=False)

print("Example excel file :")

print(pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1'))

df.to_excel('output_excel.xlsx',sheet_name='Sheet1')

#print(pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/'))

data=pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')

print(type(data))

print("Entire data : \n",data[0])
print("Head of data : \n",data[0].head())

from sqlalchemy import create_engine
engine=create_engine('sqlite:///:memory:')

data[0].to_sql('my_table',engine)
# or df.to_sql('my_table',engine)

sqldf=pd.read_sql('my_table',con=engine)
print(sqldf)
""" 
Example csv file : 

    a   b   c   d
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
Datafram from example.csv : 
     a   b   c   d
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
Example excel file :
   Unnamed: 0   a   b   c   d
0           0   0   1   2   3
1           1   4   5   6   7
2           2   8   9  10  11
3           3  12  13  14  15
<class 'list'>
Entire data : 
                          Bank NameBank       CityCity StateSt  CertCert              Acquiring InstitutionAI Closing DateClosing  FundFund
0             Heartland Tri-State Bank        Elkhart      KS     25851               Dream First Bank, N.A.       July 28, 2023     10544
1                  First Republic Bank  San Francisco      CA     59017            JPMorgan Chase Bank, N.A.         May 1, 2023     10543
2                       Signature Bank       New York      NY     57053                  Flagstar Bank, N.A.      March 12, 2023     10540
3                  Silicon Valley Bank    Santa Clara      CA     24735  First–Citizens Bank & Trust Company      March 10, 2023     10539
4                    Almena State Bank         Almena      KS     15426                          Equity Bank    October 23, 2020     10538
..                                 ...            ...     ...       ...                                  ...                 ...       ...
562                 Superior Bank, FSB       Hinsdale      IL     32646                Superior Federal, FSB       July 27, 2001      6004
563                Malta National Bank          Malta      OH      6629                    North Valley Bank         May 3, 2001      4648
564    First Alliance Bank & Trust Co.     Manchester      NH     34264  Southern New Hampshire Bank & Trust    February 2, 2001      4647
565  National State Bank of Metropolis     Metropolis      IL      3815              Banterra Bank of Marion   December 14, 2000      4646
566                   Bank of Honolulu       Honolulu      HI     21029                   Bank of the Orient    October 13, 2000      4645

[567 rows x 7 columns]
Head of data : 
               Bank NameBank       CityCity StateSt  CertCert              Acquiring InstitutionAI Closing DateClosing  FundFund
0  Heartland Tri-State Bank        Elkhart      KS     25851               Dream First Bank, N.A.       July 28, 2023     10544
1       First Republic Bank  San Francisco      CA     59017            JPMorgan Chase Bank, N.A.         May 1, 2023     10543
2            Signature Bank       New York      NY     57053                  Flagstar Bank, N.A.      March 12, 2023     10540
3       Silicon Valley Bank    Santa Clara      CA     24735  First–Citizens Bank & Trust Company      March 10, 2023     10539
4         Almena State Bank         Almena      KS     15426                          Equity Bank    October 23, 2020     10538
     index                      Bank NameBank       CityCity StateSt  CertCert              Acquiring InstitutionAI Closing DateClosing  FundFund
0        0           Heartland Tri-State Bank        Elkhart      KS     25851               Dream First Bank, N.A.       July 28, 2023     10544
1        1                First Republic Bank  San Francisco      CA     59017            JPMorgan Chase Bank, N.A.         May 1, 2023     10543
2        2                     Signature Bank       New York      NY     57053                  Flagstar Bank, N.A.      March 12, 2023     10540
3        3                Silicon Valley Bank    Santa Clara      CA     24735  First–Citizens Bank & Trust Company      March 10, 2023     10539
4        4                  Almena State Bank         Almena      KS     15426                          Equity Bank    October 23, 2020     10538
..     ...                                ...            ...     ...       ...                                  ...                 ...       ...
562    562                 Superior Bank, FSB       Hinsdale      IL     32646                Superior Federal, FSB       July 27, 2001      6004
563    563                Malta National Bank          Malta      OH      6629                    North Valley Bank         May 3, 2001      4648
564    564    First Alliance Bank & Trust Co.     Manchester      NH     34264  Southern New Hampshire Bank & Trust    February 2, 2001      4647
565    565  National State Bank of Metropolis     Metropolis      IL      3815              Banterra Bank of Marion   December 14, 2000      4646
566    566                   Bank of Honolulu       Honolulu      HI     21029                   Bank of the Orient    October 13, 2000      4645

[567 rows x 8 columns]
"""