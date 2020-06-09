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
- Whether the coach of the home team is a first year head coach
- Whether the Bleacher Report recognized the home team school as one of the top 25 tailgating destinations
- Amount of rainfall in inches
- Current snow depth in inches
- Amount it snowed during the game in inches
- Maximum temperature during the game in F
- Minimum temperature during the game in F
- Rank of the opponent (ordinal)
- Home team conference
- Date 

Some of these variables are trivially related to each other. In particular, stadium attendance and capacity define fill rate, so stadium attendance and capacity are not candidate influencing factors. Tmax and Tmin more or less move together within the short time frame of a game so Tmin defines Tmax in a sense, and vica versa, so only one of them can be considered a candidate influencing variable, and variables that are not known by potential attendees before the game cannot influence his/her attendance, so they are not candidate influencing variables in this analysis. One can further distinguish interesting variables from the ones that aren't by their p-values (the null hypothesis cannot be rejected and thus correlation cannot be evaluated). There may be interesting coupling between variables but they have not been evaluated in this analyis. Causation has also not been evaluated.

The remaining numerical variables are plotted against the fill rate, with a linear regression line and the underlying data points. Data points outside of three standard deviations above or below the mean are omitted. Each variable is accompanied by its Spearman's rank correlation coefficient. Spearman's coefficient is used throughout instead of Pearson's for generality and because the presumed most important variable is ordinal (team ranking). In the source, both coefficients are computed and do not differ by more than a few percent.

national_sr.png shows the top four Spearson's rank correlation coefficient variables (that have sufficiently small p-values) and regression lines. The strongest correlating variable is the rank of the home team (Spearson's Rho (SR) = -0.64). This supports the claim that when the home team is better (ranking is closer to the value of 1) more people go. Next is the capacity of the stadium with SR = 0.46. This supports the claim that bigger stadiums mean denser crowds. Next is the current number of losses with SR = -0.4. This supports the claim that people don't want to watch their team when they are on a losing streak. Next is opponent rank with SR = -0.25. This is a similar trend to the home team rank, but the strength of correlation is lower so it is difficult to make a meaningful statement about the relative importance of home team rank and opponent rank. correlation_matrix.png is a basic visual description of the relationship between all of the variables that confirms the a priori dinstinctions made between unintersting and interesting variables.

For reference, the data sets are from:
https://www.kaggle.com/masseyratings/rankings#cf2018.csv (the cf directory, all files)
https://www.kaggle.com/jeffgallini/college-football-attendance-2000-to-2018
