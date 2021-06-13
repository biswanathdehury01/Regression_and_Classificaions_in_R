# Count Unique values by group

# Creating dataset 
# creating class column
x <- c("A","B","C","B","A","A","C","A","B","C","A")

# creating age column
y <- c(20,15,45,14,21,22,47,18,16,50,23)

# creating age_group column
z <- c("YOUNG","KID","OLD","KID","YOUNG","YOUNG",
       "OLD","YOUNG","KID","OLD","YOUNG")

# creating dataframe
df <- data.frame(class=x,age=y,age_group=z)
df

# applying aggregate function
aggregate( age~age_group,df, function(x) length(unique(x)))