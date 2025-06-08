# Netflix Data Analysis
This project explores and analyzes the Netflix dataset to uncover insights about content trends, types, countries, and genres. It is designed for beginners in data science and Python, using commonly used libraries like pandas, matplotlib, and seaborn.
# Project Overview
The dataset netflix_titles.csv contains information about movies and TV shows available on Netflix as of 2021. The project includes various objectives that help develop skills in data cleaning, transformation, visualization, and storytelling through data.

# Objectives Covered
# 1.	Data Cleaning and Preprocessing
	•	Handled missing values using fillna()
	•	Removed duplicate entries using drop_duplicates()
	•	Converted date columns using pd.to_datetime()
# 2.	Movies vs. TV Shows Count
	•	Used countplot() to visualize the number of movies and shows (type column)
# 3.	Top Genres on Netflix
	•	Analyzed the listed_in column using value_counts() to find top 10 genres
# 4.	Content Added Over the Years
	•	Extracted year_added from the date_added column
	•	Used line plot to show how content additions changed over time
# 5.	Top Countries by Content
	•	Used value_counts() on the country column to list top contributing countries
# 6.	Most Frequent Directors
	•	Analyzed the director column to find the top 10 directors
# 7.	Monthly Trend of Content Added
	•	Extracted months from date_added and visualized monthly content additions
# 8.	Multinational Content
	•	Identified and counted content available in multiple countries (with comma-separated country names)
# 9.	Most Common Ratings
	•	Analyzed the rating column to identify the most frequent content ratings (e.g., PG, TV-MA)

⸻

# Tools & Libraries Used
	•	Python
	•	Pandas – for data manipulation
	•	Matplotlib and Seaborn – for data visualization

⸻
# How to Run
# 1.	Install the required libraries:
  ----  pip install pandas matplotlib seaborn
# 2.  Place the dataset netflix_titles.csv in the same directory as the script.
# 3.  Run the script using:
  ---- python netflix_analysis.py     

# Learnings & Skills Gained
	•	Data cleaning with missing values
	•	String operations on complex columns (like genres and countries)
	•	Time-series trend analysis
	•	Basic and advanced visualization using seaborn/matplotlib
	•	Extracting insights and presenting them clearly

# Future Improvements
	•	Add interactive visualizations using Plotly or Streamlit
	•	Group countries by continent or region
	•	Build a recommendation system prototype based on genres or directors
