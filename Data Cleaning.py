import pandas as pd

df = pd.read_csv('2023 Car Dataset.csv', encoding='ISO-8859-1')

df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

print(df.head(10))


# missing_values = df.isnull().sum()
# print(missing_values[missing_values > 0])

df['Price ($)'] = df['Price ($)'].replace({'\£': '', ',': ''}, regex=True).astype(float)
df['Customer Ratings'] = df['Customer Ratings'].str.split('/').str[0].astype(float)
df['Year'] = df['Year'].astype(int)
df['Safety Features'] = df['Safety Features'].astype(str)
df['Sales Figures (Units Sold)'] = df['Sales Figures (Units Sold)'].str.replace(',', '').astype(float)


df['Safety Features'] = df['Safety Features'].str.strip()
df['Safety Features'].fillna('Unknown', inplace=True)
print(df['Safety Features'].unique())

# df['Customer Ratings'].fillna(df['Customer Ratings'].mean(), inplace=True)
df['Sales Figures (Units Sold)'].fillna(df['Sales Figures (Units Sold)'].mean(), inplace=True)

duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

df = df.drop_duplicates()
# print((df.columns))

df.to_csv('cleaned_2023_car_dataset.csv', index=False)