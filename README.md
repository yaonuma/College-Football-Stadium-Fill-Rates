# College Football Stadium Fill Rates

The correlation of several variables with respect to the football stadium fill rate during home games is studied. The fill rate is the ratio of attendees to the capacity of the stadium. The variables included in the data set are:

- Start time of the game
- Name of opponent
- Rank of home team (averaged across all ranking methods, ordinal)
- Site of the game
- Tv channel on which the game is aired
- Result of the game
- Attendance in number of people
- Current number of wins
- Current number of losses
- Stadium capacity
- Fill rate
- Whether there is a new coach the season of the game
- Whether there was tailgating for the game
- Amount of rainfall in inches
- Current snow depth in inches
- Amount it snowed during the game in inches
- Maximum temperature during the game in F
- Minimum temperature during the game in F
- Rank of the opponent (ordinal)
- Conference to which the game belongs
- Date 

These variables are broken into two categories: numerical, categorical, and irrelevant with respect to the research question (variables must be known by the start time of the game). The numerical variables of interest are plotted against the fill rate, with a first order regression line and the underlying data points. Data points outside of three standard deviations above or below the mean are omitted. Each variable is accompanied by its Spearman's rank correlation coefficient. Spearman's coefficient is used throughout instead of Pearson's for generality and because the presumed most important variable is ordinal (team ranking). In the source, both coefficient's are computed and do not differ by more than a few percents. So the relationships are more or less linear.

Comparing the culture of the local fanbase at Michigan State University (East Lansing, MI) to the culture of the national average with Michigan State's top 4 correlating variables, it appears Michigan State football fans are pretty die hard. The fill rate is almost invariably 100%, rain or snow, hot or cold, whether their team has a good ranking, or whether the opponent has a good ranking. At the national level, this is not so. The fill rate of the national average is apporoximately 20% less than that of Michigan State's, and is more sensitive to the ranking of the home team and opponent. To gain more insight into the culture of the national average, it makes more sense to look at the top 4 correlating facors for the national average, where there are stronger correlations. The strongest correlating variable is the rank of the home team (Spearson's Rho (SR) = -0.64). When the home team is better (ranking is closer to the value of 1) more people go. The capacity of the stadium has SR = 0.46. Bigger stadiums mean denser crowds. The current number of losses has SR = -0.4. People don't want to watch their team when they are on a losing streak. The opponent rank has SR = -0.25. This is a similar trend to the home team rank, but the strength of correlation is lower. This supports the claim that people care more about their home team's rank than they care about the opponent's rank when it comes to whether they will attend the game.

There are potentially more interesting conclusions to draw with the categorical variables. These are more tedious to handle so they're not done yet. Please the source code in this repository for more information.

For reference, the data sets are from:
https://www.kaggle.com/masseyratings/rankings#cf2018.csv (the cf directory, all files)
https://www.kaggle.com/jeffgallini/college-football-attendance-2000-to-2018
