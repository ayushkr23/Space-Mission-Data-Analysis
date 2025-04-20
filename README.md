# 🚀 Space Mission Launch Analysis

A full exploratory data analysis (EDA) project using Python, designed to uncover insights from a historical dataset of global space missions.
From success rates to cost trends and country-wise contributions — this dashboard gives a 360° view of the evolution of space launches.

---

## 📁 Project Title

**Space Mission Launch Analysis**

🎓 **Course:** INT 375 – Data Science Toolbox
🏫 **Institution:** Lovely Professional University
👨‍🎓 **Submitted By:** Ayush Kumar (Reg. No. 12321643)
👩‍🏫 **Guide:** Dr. Tanima Thakur

---

## 📊 Dataset Overview

**Source:** [Maven Analytics - Space Missions Dataset](https://www.mavenanalytics.io/data-playground/dataset/space-missions/)

This dataset includes detailed launch records from 1957 to 2022, covering space agencies and companies like NASA, SpaceX, ISRO, Roscosmos, and more.

### 🔑 Key Features:

- `Company`, `Rocket`, `Location`, `Date`, `Time`
- `Price` (USD), `MissionStatus`, `RocketStatus`
- Derived fields: `LaunchDateTime`, `Country`, `Year`, `Month`, `Weekday`, `Price_Clean`

---

## 🧪 Technologies Used

| Tool / Library       | Use Case                     |
| -------------------- | ---------------------------- |
| **Python 3**   | Programming language         |
| **Pandas**     | Data cleaning & manipulation |
| **NumPy**      | Numerical operations         |
| **Matplotlib** | Data visualization           |
| **Seaborn**    | Statistical plotting         |

---

## 🔍 Project Objectives

1. **Success Rate Analysis**

   - Compare mission outcomes across top countries & companies
2. **Price vs Outcome Analysis**

   - Explore cost patterns for successful vs failed missions
3. **Temporal Trends**

   - Weekly and monthly launch frequency patterns
4. **Mission Status Over Time**

   - Analyze how launch outcomes have changed from 1957 to 2022
5. **Country-wise Launch Distribution**

   - Identify top countries by number of launches
6. **Correlation Analysis**

   - Study relationship between `Price`, `Year`, and `Month`

---

## 📈 Visual Insights

All visualizations were created using `matplotlib` and `seaborn`, including:

- Bar plots of top companies and countries
- Stacked bar charts comparing success vs failure
- Line plots showing price trends over the years
- Box plots for price distribution by mission outcome
- Correlation heatmaps showing numeric relationships
- Count plots for launch frequency by month and weekday
- Year-wise breakdown of mission statuses

📂 Saved output plots:

- `space_mission_overview.png`
- `success_by_country_company.png`
- `price_vs_status_boxplot.png`
- `temporal_trends.png`
- `price_trend.png`
- `country_distribution.png`
- `correlation_heatmap.png`
- `mission_status_over_time.png`

---

## 📘 Key Findings

- The US and Russia dominated space activity historically; China and India have seen recent growth.
- Companies like RVSN USSR and CASC have conducted the most missions.
- Private players like SpaceX have high success rates with frequent launches in recent years.
- Higher-cost missions tend to succeed more, but success is not solely dependent on budget.
- Clear patterns in launch timing: more missions in mid-year months and certain weekdays.

---

## 🔮 Future Scope

- Add recent mission data (post-2022)
- Use machine learning to predict mission success
- Include mission payload, orbit type, and purpose for deeper analysis
- Build an interactive dashboard using Streamlit or Plotly Dash
- Explore the environmental impact of launches (fuel, emissions)

---

## 📌 How to Run the Code

1. Clone this repo:

```bash
git clone https://github.com/yourusername/space-mission-analysis.git
cd space-mission-analysis
```
