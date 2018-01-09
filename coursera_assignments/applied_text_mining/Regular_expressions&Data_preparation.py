'''Assignment 1¶
In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data.

Each line of the dates.txt file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.

The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates.

Here is a list of some of the variants you might encounter in this dataset:

04/20/2009; 04/20/09; 4/20/09; 4/3/09
Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
Feb 2009; Sep 2009; Oct 2010
6/2008; 12/2009
2009; 2010
Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:

Assume all dates in xx/xx/xx format are mm/dd/yy
Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
Watch out for potential typos as this is a raw, real-life derived dataset.
With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.

For example if the original series was this:

0    1999
1    2010
2    1978
3    2015
4    1985
Your function should return this:

0    2
1    4
2    0
3    1
4    3
Your score will be calculated using Kendall's tau, a correlation measure for ordinal data.

This function should return a Series of length 500 and dtype int.'''

###My solution

import pandas as pd
import re

#calculate month-mapping dictionaries
import calendar
month_abbr = (dict((v,k) for k,v in enumerate(calendar.month_abbr)))
months = (dict((v,k) for k,v in enumerate(calendar.month_name)))

#Create Test data frame for testing
def create_test_data_frame():
    test_dates = ['04/20/2009', '04/20/09', '4/20/09', '4/3/09', 'Mar-20-2009', 'Mar 20, 2009', 
                  'March 20, 2009', 'Mar. 20, 2009', 'Mar 20 2009', '20 Mar 2009', '20 March 2009',
                  '20 Mar. 2009', '20 March, 2009', 'Mar 20th, 2009', 'Mar 21st, 2009', 'Mar 22nd, 2009',
                  'Feb 2009', 'Sep 2009', 'Oct 2010', '6/2008', '12/2009', ' 2009 ', ' 2010 ']
    check_line = (len(test_dates) * [False])
    merged = pd.DataFrame(
        {'data': test_dates,
         'check_line': check_line
        })
    return pd.DataFrame(merged)

#Open Data
def open_data():
    doc = []
    with open('dates.txt') as file:
        for line in file:
            line = line.strip()
            doc.append(line)
    check_line = (len(doc) * [False])
    merged = pd.DataFrame({'data': doc,'check_line': check_line})
    return merged

#1 Replace all dates with standard formats: <<<month-day-year>>>
def replace_data_to_standard_format(data):
    #Replace all formats except single year format
    data['formatted'] = replace_old_formats_with_re(data['data'])
    data = check_if_worked_on(data)
    data['original_rank']=data.index
    
    #Split data to manipulate single year format
    data1 = data[data['check_line']==True].copy()
    data2 = data[data['check_line']==False].copy()
    data2['formatted'] = replace_single_years(data2['formatted'])
    data2['check_line'] = check_if_worked_on(data2)
    data_new = pd.merge(data1, data2, how='outer')
    data_new = data_new.sort_values('original_rank')
    return data_new

def replace_old_formats_with_re(df):
    # 1.1 04/20/2009; 04/20/09; 4/20/09; 4/3/09
    df = df.str.replace(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})\b', r'<<<\1-\2-\3>>>') #04/20/09
    df = df.str.replace(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', r'<<<\1-\2-\3>>>')
    
    # 1.2 Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
    df = df.str.replace(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-. ]*(\d{1,2})[, -]*(\d{4})', r'<<<\1-\2-\3>>>')
    
    # 1.3 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
    df = df.str.replace(r'(\d{1,2})[ ]*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[,. ]*(\d{4})', r'<<<\2-\1-\3>>>')
    
    # 1.4 Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
    df = df.str.replace(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ ]*(\d{1,2})(st|nd|rd|th)[, ]*(\d{4})', r'<<<\1-\2-\4>>>')
    
    # 1.5 Feb 2009; Sep 2009; Oct 2010, August 2008,
    df = df.str.replace(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[, ]*(\d{4})', r'<<<\1-'+'01-'+r'\2>>>')
    
    # 1.6 6/2008; 12/2009
    df = df.str.replace(r'^(\d{1,2})[ /]+(\d{4})', r'<<<'r'\1-'+'01-'+r'\2>>>')
    df = df.str.replace(r'\D(\d{1,2})[ /]+(\d{4})', r'<<<'r'\1-'+'01-'+r'\2>>>')
    return df

def check_if_worked_on(df): 
    df['check_line']=df['formatted'].str.contains('>>>')
    return df

def replace_single_years(df): 
    # 1.7 2009; 2010
    df = df.str.replace(r'\D(\d{4})\D', r'<<<01-'+'01-'+r'\1>>>')
    df = df.str.replace(r'^(\d{4})\D', r'<<<01-'+'01-'+r'\1>>>')
    df = df.str.replace(r'\D(\d{4})$', r'<<<01-'+'01-'+r'\1>>>')
    return df

def replace_years_that_are_left(df):
    df = df.str.replace(r'(\d{4})', r'<<<01-'+'01-'+r'\1>>>')
    return df
    
    
##2 Change into mm--dd-yyyy format and copy into new column
def std_format(df):
    from datetime import datetime
    dates=[]
    #Getting dates and transforming into final format with loop
    for index, row in enumerate(df['formatted']):
        #setting positions in string
        start = row.find('<<<')+3
        end = row.find('>>>')
        md = row.find('-', start)+1
        dy = row.find('-', md)+1
        #getting information
        month = str(row[start:md-1])
        day = str(row[md:dy-1])
        year = str(row[dy:end])

        # transforming day
        if len(day) == 1:
            day = '0' + day
        # transforming year
        if len(year) == 2:
            year = '19' + year
        # transforming month
        if len(month) == 3:
            month = str(month_abbr[month])
        elif len(month) > 3:
            month = months[month]
        if len(month) == 1:
            month = '0' + month
        else:
            month = month
        date = month+'-'+day+'-'+year
        date=datetime.strptime(date, '%m-%d-%Y')
        dates.append(date)
    
    df['dates'] = pd.Series(dates)
    return df['dates']


##3 Main functions for production & test vector; sorting data as requested
def date_sorter():
    data = open_data()
    data['original_rank'] = data.index
    df = replace_data_to_standard_format(data)
    df = df.sort_values('original_rank')
    df['dates'] = std_format(df)
    df['rank'] = df['dates'].rank(method='first')
    df['rank'] = df['rank'] - 1
    df['rank'] = df['rank'].astype(int)
    df = df.set_index('rank')
    df = df.sort_index()
    return df['original_rank']

def test_date_sorter():
    data = create_test_data_frame()
    data['original_rank'] = data.index
    df = replace_data_to_standard_format(data)
    df = df.sort_values('original_rank')
    df['dates'] = std_format(df)
    df['rank'] = df['dates'].rank(method='first')
    df['rank'] = df['rank'] - 1
    df['rank'] = df['rank'].astype(int)
    df = df.set_index('rank')
    df = df.sort_index()
    return df

