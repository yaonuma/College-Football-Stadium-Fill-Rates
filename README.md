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

A priori, it is useful to make a distinction between interesting variables and uninteresting ones. Uninteresting variables are trivially related to other variables by either common sense or mathematical definiton. In particular, stadium attendance and capacity define fill rate, so stadium attendance and capacity are not candidate influencing factors. Tmax and Tmin more or less move together within the short time frame of a game so Tmin defines Tmax in a sense, and vica versa, so only one of them can be considered a candidate influencing variable, and variables that are not known by potential attendees before the game cannot influence his/her attendance, so they are not candidate influencing variables in this analysis. Posteriori, it is necessary to distinguish uninteresting variables from interesting ones by the p-value (the null hypothesis cannot be rejected and thus correlation cannot be evaluated). There may be interesting nonlinearly coupled variables but they have not been evaluated in this analyis. Causation has also not been evaluated.

The remaining interesting numerical variables are plotted against the fill rate, with a linear regression line and the underlying data points. Data points outside of three standard deviations above or below the mean are omitted. Each variable is accompanied by its Spearman's rank correlation coefficient. Spearman's coefficient is used throughout instead of Pearson's for generality and because the presumed most important variable is ordinal (team ranking). In the source, both coefficients are computed and do not differ by more than a few percent.

For reference, the data sets are from:
https://www.kaggle.com/masseyratings/rankings#cf2018.csv (the cf directory, all files)
https://www.kaggle.com/jeffgallini/college-football-attendance-2000-to-2018
