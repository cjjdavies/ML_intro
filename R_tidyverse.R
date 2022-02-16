### R as a calculator
  
# In the Console window below, type: 2+2 and press Enter
# Also try:

2^5
3/10
(3+5)^2
sqrt(4)

# Tip: To run a line (or multiple lines) of code from a script without typing them into the Console, 
# select the line(s) you want to run and press Ctrl+Enter (Command+Enter on a Mac)

### Objects

# R stores data as objects. You create new objects when you assign a value to them using "<-":
  
x <- 3  # Check the "Environment" window!


# Tip: use the R studio shortcut Alt+ - (Alt and the minus sign) to easily create the assignment symbol <-

y <- 6
x+y

# Tip: R is case sensitive so if you've defined your object as x, it will not recognise (capital) X. 
# Similarly, the function for square root is sqrt, R will give you an error if you try to use Sqrt.


### Packages

# Packages extend the functionality of base R. They are distributed via CRAN: the Comprehensive R Archive Network

# To install a package, use: install.packages("packagename")

# You then need to load it, using library(packagename)

# We will be using a collection of packages called the Tidyverse:

library(tidyverse)

# When you load the tidyverse, you'll see a message about conflicts. As there is an (increasingly) large number of packages in R, it is possible to have functions with the same name in more than one package. The message tells you that packages dplyr and stats both have a function called filter and the one that will be used is the one from dplyr. It is the one that was loaded last.

# If you want to use a function from a particular package, you need to include packagename:: before the name of the function.

# In this example, you can use stats::filter() instead of just filter() to use filter from the stats package.

find("filter") # this shows you the packages a function belongs to, in order of priority

# The tidyverse packages we will be using mostly are readr (for reading in data), dplyr (for transforming data) and ggplot2 (for plotting). 

### Functions

# When using the Tidyverse, you can call functions in two ways:
  
sqrt(4) # base R

4 %>%
  sqrt  # "pipe" operator (you can read is as "and then...")

# Tip: use the R Studio shortcut Ctrl + Sft + M to create the pipe operator %>%
  
# Tip: If you are not sure what a function does, type ?functionname in the Console:
  
?sqrt

### Reading in data

# To read in a comma-separated value (csv) file, use the read_csv function:
  
fev_data <- read_csv("fev.csv")

# Tip: If you got an error that "fev.csv" does not exist, check that you are working in the correct directory!
  
# Look at the top few rows of the data frame:
  
head(fev_data)

# Tip: if your data is in a Microsoft Excel spreasheet, you will need a different package to read it in, such as readxl. So you'll need:

install.packages("readxl")
library(readxl)
excel_data <- read_xlsx(filename, sheet = 1) #(to read the first sheet)

# Apply the summary function to fev_data. What does it do?

# Add your code here!
summary(fev_data)

# The summary function provides summary statistics for each of the variables

### Operating on data: columnns

# Individual columns are identified using the $ symbol:

head(fev_data$fev)
summary(fev_data$fev)
length(fev_data$fev)

# Other useful functions for data frames:

names(fev_data)
dim(fev_data)

# Other useful functions for columns:

max(fev_data$fev)
mean(fev_data$fev)
sd(fev_data$fev)

### Operating on data: subsets

# To select subsets of the data (not just columns with $) use square brackets:

fev_data$fev[32] # 32nd element of the fev column

fev_data[32,3] # 32nd element of the 3rd column

fev_data[32,"age"]  # Same thing, but using the name of the 3rd column - better, as it is more readable and robust

fev_data[32, ] # Everything in the 3rd row

fev_data[32,1:3]

fev_data[32,-5]

fev_data[32,-1:-2]

fev_data[32,c(1,3,5)] #c(1,3,5) is a vector of numbers (c means "combine")

c(1,3,5) %>%
  length

# How would you drop the 1st, 3rd and 5th column?

# Add your code here!

fev_data_dropped <- fev_data[-c(1, 3, 5)]

### Logicals

# Besides numbers and strings of characters, R also stores logicals - TRUE and FALSE

# Example: a new vector with elements that are TRUE if height is above 72 cm and FALSE otherwise:

is_tall <- fev_data$height > 72

# Useful summary command:

table(is_tall)

# Which subjects in fev_data are tall?

fev_data[is_tall,]

### Filtering (selecting rows)

fev_data %>%
  filter(height > 72)

fev_data %>%
  filter(age == 6)

fev_data %>%
  filter(age != 20)

fev_data %>%
  filter(age <= 20)

# You can also filter by whether data are not a number (na):

fev_data %>%
  filter(is.na(age))  # opposite: !is.na(age)

# You can combine multiple expressions with Boolean operators: & is “and”, | is “or”, and ! is “not”

fev_data %>%
  filter(age == 14 & smoke !=0)   # age is 14 AND smoker

fev_data %>%
  filter(age < 5 | height < 50)  # younger than 5 OR shorter than 50 cm

# Rules for filtering for categorical data:
sex == "F" or sex != "F"
sex %in% c("M","F")

### Selecting columns

fev_data %>%
  select(fev, height, age)

fev_data %>%
  select(-seqnbr, -subjid)

### Summarising data

fev_data %>%
  filter(age == 14 & smoke != 0) %>%
  summarise(mean(fev))

# You can name the summary variable:

fev_data %>%
  filter(age == 14 & smoke != 0) %>%
  summarise(my_mean = mean(fev))


fev_data %>%
  filter(age == 14 & smoke != 0) %>%
  summarise(mean(fev), sd(fev))

# To get the average FEV for both smokers and non-smokers we don't need to repeat for smoke==0. We can create a grouping variable:
  
fev_data %>%
  group_by(smoke)

# (Same exact data, it just prints the two groups)

fev_data %>%
  group_by(smoke) %>%
  summarise(mean(fev), sd(fev))

# But what is the size of each group? n() gives us the number of observations in each group:
  
fev_data %>%
  group_by(smoke) %>%
  summarise(n = n(), mean = mean(fev), sd = sd(fev))

# You can also group by your own variables:
  
fev_data %>%
  group_by(height < 60) %>%
  summarise(n(), mean(fev))

# A useful function: arrange

fev_data %>%
  group_by(age) %>%
  summarise(n(), mean(fev)) %>%
  arrange(age) # arrange by increasing age

fev_data %>%
  group_by(age) %>%
  summarise(n(), mean(fev)) %>%
  arrange(desc(age)) # arrange by decreasing age

# Sorting columns

fev_data$age %>% sort #Sort a column

fev_data$age %>% unique %>% sort # Sort unique values in a column

# table() gives you a count of a particular factor or combination of factor levels:
  
table(fev_data$age)
table(fev_data$age,fev_data$smoke)

# **Problem 1**: Which subjects are male and which are female? (i.e. what does sex == 1 mean?) If this was a real dataset, we would expect to have this information, but let's pretend we don't. Are there any variables in the datasets we could use to infer this?
  
# Add your code here!

fev_data %>%
  group_by(age,sex) %>%
  summarise(n(),mean(height), sd(height)) %>%
  arrange(desc(age))

# Perhaps 0 = Female and 1 = Male because males tend to be taller than females?

# **Problem 2**: Why do smokers appear to have better lung function (higher forced expiratory volume - FEV)?
  
# Add your code here!

fev_data %>%
  group_by(smoke) %>%
  summarise(mean(fev))

fev_data %>%
  group_by(age,smoke) %>%
  summarise(n = n(), mean = mean(fev), sd = sd(fev)) %>%
  filter(n>3) %>%
  arrange(desc(age))

# Useful function: rename

fev_data %>%
  rename(ID = subjid)

# Mutate: compute new column

fev_data %>%
  mutate(heightdiff = height - mean(height))

# Remember that to save these changes you need to assign to a new data frame:
  
new_fev_data <- fev_data %>%
  rename(id = subjid) %>%
  mutate(heightdiff = height - mean(height))

### Plotting

# The first step to using ggplot is creating a blank canvas:
  
fev_data %>%
  ggplot(aes(x = age, y = fev))   # aes stands for "aesthetic"

### Scatterplots

# We add points using geom_point:
  
fev_data %>%
  ggplot(aes(x = age, y = fev)) +
  geom_point()

# Add limits, labels, title:
  
fev_data %>%
  ggplot(aes(x = age, y = fev)) +
  geom_point() +
  ylim(0, 7) +
  ylab("Forced exhalation\nvolume (litres)") +
  ggtitle("Distribution of FEV")

# We can change the colour of the points (colour - but you can also use col or color), the shape of the points (shape) and the size in mm (size):
  
fev_data %>%
  ggplot(aes(x = age, y = fev)) +
  geom_point(size = 0.5, shape = 18, colour = "blue")  # Shape 18 is filled diamond

# Tip: Find out more about ggplot's aesthetic specifications here: https://cran.r-project.org/web/packages/ggplot2/vignettes/ggplot2-specs.html

# To colour by sex, we add the variable as aesthetic element:

fev_data %>%
  mutate(sex = as.character(sex)) %>% # change sex from continuous to categorical variable
  ggplot(aes(x = age, y = fev)) +
  geom_point(aes(colour = sex))

# Add a smooth line fitted to the data:

fev_data %>%
  mutate(sex = as.character(sex)) %>%
  ggplot(aes(x = age, y = fev)) +
  geom_point(aes(colour = sex)) +
  geom_smooth()

# One line per sex:

fev_data %>%
  mutate(sex = as.character(sex)) %>%
  ggplot(aes(x = age, y = fev)) +
  geom_point(aes(colour = sex)) +
  geom_smooth(aes(colour=sex))

### Line plots

fev_average <- fev_data %>%
  group_by(age, smoke) %>%
  summarise(fev_mean = mean(fev))

# This is not quite right:

fev_average %>%
  ggplot(aes(x = age, y = fev_mean)) +
  geom_line(aes(colour = smoke))  # smoke is a continuous numeric variable!

# This is better :)

fev_average %>%
  mutate(smoke = as.character(smoke)) %>%
  ggplot(aes(x = age, y = fev_mean)) +
  geom_line(aes(colour = smoke))

### Bar plots

# How many subjects smoke?

fev_data %>%
  mutate(smoke = as.character(smoke)) %>%
  ggplot(aes(x = smoke)) + 
  geom_bar()  # by default, the y variable is the count of the x (in this case, number of smokers)

# How many men and how many women smoke / don't smoke?
  
fev_count <- fev_data %>%
  group_by(sex, smoke) %>%
  summarise(smoke_count = n())

# It might be easier to replace 0 and 1 with meaningful words...

fev_count <- fev_data %>%
  mutate(Smoke = ifelse(smoke == 1, "Smoker", "Non-smoker"),
         Sex = ifelse(sex == 1, "Male", "Female")) %>%
  group_by(Sex, Smoke) %>%
  summarise(Count = n())

fev_count %>%
  ggplot(aes(x = Smoke, y = Count, fill = Sex)) + 
  geom_bar(stat = "identity",     # "identity" allows us to specify a y variable, otherwise it is count (as above)
           position = "dodge") +  # Put bars side-by-side instead of stacked
  scale_fill_brewer(palette="Spectral")

# Tip: Find out much more about colours in ggplot here: http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/

### Box plots
  
fev_data %>%
  filter(age < 18 & age > 9) %>%
  mutate(smoke = as.character(smoke),
         age = as.character(age)) %>%
  ggplot(aes(x = age, y = fev, colour = smoke)) +
  geom_boxplot() + 
  theme_bw()