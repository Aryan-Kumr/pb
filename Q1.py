
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset
data = {
    'plant_name': ['Fern', 'Cactus', 'Bamboo', 'Rose', 'Tulip', 'Daisy', 'Sunflower', 'Lily', 'Orchid', 'Maple'],
    'sunlight_exposure': [5, 10, 12, 8, 6, 7, 14, 11, 9, 5],  # hours of sunlight
    'plant_height': [30, 150, 200, 60, 50, 40, 180, 70, 40, 20]  # height in cm
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Step 2: Visualize the relationship using a scatterplot
plt.scatter(df['sunlight_exposure'], df['plant_height'], color="black", marker="*")
plt.title('Relationship between Sunlight Exposure and Plant Height')
plt.xlabel('Sunlight Exposure (hours)')
plt.ylabel('Plant Height (cm)')
plt.grid(True)
plt.show()

# Step 3: Calculate the correlation coefficient
correlation = df['sunlight_exposure'].corr(df['plant_height'])
print(f"Correlation between sunlight exposure and plant height: {correlation}")

# Step 4: Determine if the association is significant
threshold = 0.7
if abs(correlation) >= threshold:
    print("There is a significant association between sunlight exposure and plant growth rate.")
else:
    print("There is no significant association between sunlight exposure and plant growth rate.")
