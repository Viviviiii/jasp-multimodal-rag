# USING THE JASP ENVIRONMENT (page 6)


Open JASP.
The main menu can be accessed by clicking on the top-left icon.
Open:
JASP has its own . jasp format but can open a variety of
different dataset formats, such as:
.xls/xlsx (Excel files)
.csv (comma-separated values) can be saved in Excel
.txt (plain text) can also be saved in Excel
.tsv (tab-separated values) can also be saved in Excel
.sav (IBM SPSS data file)
.ods (Open Document spreadsheet)
.dta (Stata data file)
.por (SPSS ASCII file)
.Sas7bdat /cat (SAS data files)
.xpt (SAS transport file)
You can open recent files, browse your computer files, access the Open Scien ce Framework (OSF) or
open the wide range of examples that are packaged with the Data Library in JASP.

Save/Save as:
Using these options for the data file, any annotations and the analysis
can be saved in the .jasp format
Export:
Results can be exported to either an HTML file or a PDF file
Data can be exported to either a .csv, .tsv or .txt file
Sync data:
Used to synchroni se with any updates in the current data file (also
can use Ctrl-Y)
Close:
As it states, it closes the current file but not JASP.
Preferences:
There are four sections that users can use to tweak JASP to suit their needs
In the Data Preferences section, users can:
Set the default spreadsheet editor (i.e. Excel, SPSS, etc.)
Change the threshold so that JASP more readily distinguishes between nominal and scale data.
Add a custom missing value code.

In the Results Preferences section, users can:
Set JASP to return exact p values, i.e. P=0.00087 rather than P<.001
Fix the number of decimals for data in tables  makes tables easier to read/publish
Change the pixel resolution of the graph plots
Select when copying graphs whether they have a white or transparent background.
PDF output settings
In the Interface Preferences section, users can now define a user font and pick between two different
themes: a light theme (default) and a dark theme.
The preferred language currently supports 13 languages: English, Spanish, German, Dutch, French,
Indonesian, Japanese, Portuguese, Chinese and Galician.
In this section, there is also the ability to change the system size  (zoom) for accessibility and scroll
speeds.

Comparison of the dark and light themes in JASP
In the Advanced Preferences section, most users will probably never have to change any of the default
settings.

JASP has a streamlined interface to switch between the spreadsheet, analysis and results views.
The vertical bars highlighted above allow for the windows to be dragged right or left by clicking and
dragging the three vertical dots.
The individual windows can also be completely collapsed using the right or left arrow icons.
If you click the  Results      icon, a range of options is provided, including:
Edit title
Copy
Export results
Add notes
Remove all
Refresh all
The add notes option allows the results output to be easily annotated and then exported to an HTML
or PDF file by going to File > Export Results.

The Add notes menu provides many options to change text font, colour, size, etc.
You can change the size of all the tables and graphs using Ctrl+ (increase), Ctrl- (decrease), Ctrl=
(back to default size). Graphs can also be resized by dragging the bottom right corner of the graph.
As previously mentioned, all tables and figures are APA standard and can just be copied into any other
document. Since all images can be copied/saved with either a white or transparent background. This
can be selected in  Preferences > Advanced as described earlier.
There are many further resources on using JASP on the website https://jasp-stats.org/

# USING THE JASP ENVIRONMENT (page 6)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_USING_THE_JASP_ENVIRONMENT_PAGE_6]

# DATA HANDLING IN JASP (page 12)


For this section, open England injuries.csv
All files must have a header label in the first row.  Once loaded, the dataset appears in the window:
For large datasets, there is a hand icon that allows easy scrolling through the data.
On import, JASP makes a best guess at assigning data to the different variable types:
Nominal                                 Ordinal                                       Continuous
If JASP has incorrectly identified the data type, just click on the appropriate variable data icon in the
column title to change it to the correct format.
If you have coded the data, you can click on the variable name to open the following window in which
you can label each code. These labels now replace the codes in the spreadsheet view. If you save this
as a .jasp file, these codes, as well as all analyses and notes,  will be saved automatically. This makes
the data analysis fully reproducible.

In this window, you can also carry out simple filtering of data, for example, if you untick the Wales
label, it will not be used in subsequent analyses.
Clicking this icon in the spreadsheet window opens up a much more comprehensive set of data
filtering options:
Using this option will not be covered in this document. For detailed information on using more
complex filters, refer to the following link: https://jasp-stats.org/2018/06/27/how-to-filter-your-data-
in-jasp/

By default, JASP plots data in the Value order (i.e. 1-4). The order can be changed by highlighting the
label and moving it up or down using the appropriate arrows:
Move up
Move down
Reverse order
Close
If you need to edit the data in the spreadsheet , just double-click on a cell and the data should open
up in the original spreadsheet , i.e. Excel. Once you have edited your data and saved the original
spreadsheet, JASP will automatically update to reflect the changes that were made, provided that you
have not changed the file name.

# DATA HANDLING IN JASP (page 12)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_DATA_HANDLING_IN_JASP_PAGE_12]

# DATA EDITING IN JASP (page 15)


JASP has its internal data editor, which allows users to either enter new data directly,
or to edit existing (external) data.
Clicking on this opens a new ribbon allowing the user to add new data directly or edit an
open dataset.  If an external dataset has been modified, the Synchronisation option will save
those changes to the dataset directly.
Once the editing has been completed, you can return directly to the data analysis interface.

# DATA EDITING IN JASP (page 15)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_DATA_EDITING_IN_JASP_PAGE_15]

# JASP ANALYSIS MENU (page 16)


The main analysis options can be accessed from the main toolbar. Currently, JASP offers the following
frequentist (parametric and non-parametric standard statistics) and alternative Bayesian tests:
Descriptives
Descriptive stats
Raincloud plots
Time Series Descriptives
Flexplot
Regression
Correlation
Linear regression
Logistic regression
Generalised Linear Model*
T-Tests
Independent
Paired
One sample
Frequencies
Binomial test
Multinomial test
Contingency tables
Log-linear regression*
ANOVA
Independent
Repeated measures

# JASP ANALYSIS MENU (page 16)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_JASP_ANALYSIS_MENU_PAGE_16]

# • ANCOVA: • MANOVA * (page 16)


Factor
Principal Component Analysis (PCA)*
Exploratory Factor Analysis (EFA)*
Confirmatory Factor Analysis (CFA)*
Mixed Models*
Linear Mixed Models
Generalised linear mixed models
Meta-Analysis
Effect size computation
Meat-Analysis
Forest Plots
* Not covered in this guide
BY clicking on the + icon on the top-right menu bar, you can also access advanced options that allow
the addition of optional advanced modules. Once ticked , they will be added to the main analysis
ribbon.
See the JASP website for more information on these advanced modules.

Once you have selected your required analysis , all the possible statistical options appear in the left
window and the output appears in the right window.
JASP provides the ability to rename and stack the results output, thereby organising multiple
analyses.
The individual analyses can be renamed using the pen icon or deleted using the red cross.
Clicking on the analysis in this list will then take you to the appropriate part of the results output
window. They can also be rearranged by dragging and dropping each of the analyses.
The green + icon produces a copy of the chosen analysis.
The blue information icon provide s detailed information on each of the statistical procedures used
and includes a search option.

# • ANCOVA: • MANOVA * (page 16)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_ANCOVA_MANOVA_PAGE_16]

# DESCRIPTIVE STATISTICS (page 19)


Presentation of all the raw data is very difficult for a reader to visualise or draw any inference from.
Descriptive statistics and related plots are a succinct way of describing and summarising data, but do
not test any hypotheses. Various types of statistics are used to describe data:
Measures of central tendency
Measures of dispersion
Percentile values
Measures of distribution
Descriptive plots
To explore these measures, load Descriptive data.csv  into JASP. Go to Descriptives > Descriptive
statistics and move the Variable data to the Variables box on the right.
You also have options to change and add tables in this section:
Split analyses by a categorical variable (i.e., group)
Transpose the main descriptive table (switch columns and rows)

The Statistics menu can now be opened to see the various options available.

# DESCRIPTIVE STATISTICS (page 19)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_DESCRIPTIVE_STATISTICS_PAGE_19]

# CENTRAL TENDENCY. (page 20)


This can be defined as the tendency for variable values to cluster around a central value. The three
ways of describing this central value are mean, median and mode. If the whole population is
considered, the term population mean / median/mode is used. If a sample/subset of the population
is being analysed, the term sample mean/ median/mode is used. The measures of central tendency
move toward  a constant value when the sample size is sufficient to be representative of  the
population.

The mean, M or x (17.71), is equal to the sum of all the values divided by the number of values in the
dataset, i.e. the average of the values. It is used for describing continuous data. It provides a simple
statistical model of the centre of distribution of the values and is a theoretical estimate of the typical
value. However, it can be influenced heavily by extreme scores.
The median, Mdn (17.9), is the middle value in a dataset that has been ordered from the smallest to
the largest value and is the normal measure used for ordinal or non-parametric continuous data. Less
sensitive to outliers and skewed data
The mode (20.0) is the most frequent value in the dataset and is usually the highest bar in a distribution
histogram.
DISPERSION
The standard error of the mean, SE  (0.244), is a measure of how far the sample mean of the data is
expected to be from the true population mean. As the size of the sample data grows larger , the SE
decreases compared to S, and the true mean of the population is known with greater specificity.
Standard deviation, S or SD (6.935), is used to quantify the amount of dispersion of data values around
the mean. A low standard deviation indicates that the values are close to the mean, while a high
standard deviation indicates that the values are dispersed over a wider range.

The coefficient of variation  (0.392) provides the relative dispersion of the data, in contrast to the
standard deviation, which gives the absolute dispersion.
MAD, (4.7) median absolute deviation, a robust measure of the spread of data. It is relatively
unaffected by data that is not normally distributed. Reporting median +/ - MAD for data that is not
normally distributed is equivalent to mean +/- SD for normally distributed data.
MAD Robust : (6.968) m edian absolute deviation of the data points, adjusted by a factor for
asymptotically normal consistency.
IQR (9.175)  Interquartile Range is similar to the MAD but is less robust (see Boxplots).
Variance (48.1) is another estimate of how far the data is spread from the mean. It is also the square
of the standard deviation.
Confidence intervals (CI), although not shown in the general Descriptive statistics output, are used in
many other statistical tests. When sampling from a population to get an estimate of the mean,
confidence intervals are a range of values within which you are n% confident the true  mean is
included. A 95% CI is, therefore, a range of values that one can be 95% certain contains the true mean
of the population. This is not the same as a range that contains 95% of ALL the values.
For example, in a normal distribution, 95% of the data are expected to be within  1.96 SD of the mean
and 99% within  2.576 SD.
95% CI = M  1.96 * the standard error of the mean.
Based on the data so far, M = 17.71, SE = 0.24; this will be 17.71  (1.96 * 0.24) or 17.71  0.47.
Therefore, the 95% CI for this dataset is 17.24 - 18.18 and suggests that the true mean is likely to be
within this range 95% of the time
QUARTILES
In the Statistics options, make sure that everything is unticked apart from Quartiles.

Quartiles are where datasets are split into 4 equal quarters, normally based on the rank ordering of
median values. For example, in this dataset
1 1 2 2 3 3 4 4 4 4 5 5 5 6 7 8 8 9 10 10 10
25%     50%     75%
The median value that splits the data by 50% = 50th percentile = 5
The median value of left side = 25th percentile = 3
The median value of right side = 75th percentile = 8
From this, the Interquartile range (IQR) can be calculated, which is the difference between the 75th
and 25th percentiles, i.e. 5.  These values are used to construct the descriptive box  plots later.  The
IQR can also be shown by ticking this option in the Dispersion menu.
DISTRIBUTION
Skewness describes the shift of the distribution away from a normal distribution. Negative skewness
shows that the mode moves to the right, resulting in a dominant left tail. Positive skewness shows
that the mode moves to the left, resulting in a dominant right tail.
Negative skewness
Positive skewness

Kurtosis describes how heavy or light the tails are. Positive kurtosis results in an increase in the
pointiness of the distribution with heavy (longer) tails, while negative kurtosis exhibits a much more
uniform or flatter distribution with light (shorter) tails.
In the Statistics options, make sure that everything is unticked apart from skewness, kurtosis and the
Shapiro-Wilk test.
We can use the Descriptives output to calculate skewness and kurtosis. For a normal data distribution,
both values should be close to zero. The Shapiro -Wilk test is used to assess whether the da ta is
significantly different f rom a normal distribution.  (see - Exploring data integrity in JASP for more
details).
INFERENCE
Confidence intervals provide a measure of how the data sample represents the population being
studied. The probability that the confidence interval includes the true mean of the population is
termed the confidence level and is typically set at 95%.
+ kurtosis
Normal
- kurtosis

# CENTRAL TENDENCY. (page 20)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_CENTRAL_TENDENCY_PAGE_20]

# SPLITTING DATA FILES (page 25)


If there is a grouping variable (categorical or ordinal), descriptive statistics and plots can be produced
for each group. Using Descriptive data.csv with the variable data in the Variables box, now add Group
to the Split box.

# SPLITTING DATA FILES (page 25)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_SPLITTING_DATA_FILES_PAGE_25]

# DESCRIPTIVE DATA VISUALISATION (page 25)


JASP produces a comprehensive range of descriptive and analysis-specific plots. These plots will be
explained in their relevant chapters.

# DESCRIPTIVE DATA VISUALISATION (page 25)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_DESCRIPTIVE_DATA_VISUALISATION_PAGE_25]

# BASIC PLOTS (page 25)


Firstly, to look at examples of the basic plots, open Descriptive data.csv with the variable data in the
Variables box, go to Plots and tick Distribution plots, Display density, Interval plots, Q-Q plots, and dot
plots.

The Distribution plot is based on splitting the data into frequency bins, which are then overlaid with
the distribution curve. As mentioned before, the highest bar is the mode (the most frequent value of
the dataset. In this case, the curve looks approximately symmetrical , suggesting that the data is
approximately normally distributed. The second distribution plot is from another dataset , which
shows that the data is positively skewed.
The dot plot displays the distribution where each dot represents a value. If a value occurs more than
once, the dots are placed one above the other so that the height of the column of dots represents the
frequency for that value.
The interval plot shows a 95% confidence interval for the mean of each variable.
The Q-Q plot (quantile-quantile plot) can be used to visually assess if a set of data comes from a normal
distribution. Q -Q plots take the sample data, sort it in ascending order, and then plot  it against
quantiles (percentiles) calculated from a theoretical distribution. If the data is normally distributed,
the points will fall on or close to the 45-degree reference line. If the data is not normally distributed,
the points will deviate from the reference line.

Depending on the data sets, basic correlation graphs and pie charts for non-scale data can also be
produced, as can Pareto and Likert plots.

# BASIC PLOTS (page 25)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_BASIC_PLOTS_PAGE_25]

# CUSTOMISABLE PLOTS (page 28)


There are a variety of options depending on your datasets.
The boxplots visualise several statistics described above in one plot:
Median value
25 and 75% quartiles
Interquartile range (IQR) i.e., 75% - 25% quartile values
Maximum and minimum values plotted with outliers excluded
Outliers are shown if requested

Go back to the statistics options, in Descriptive plots , tick both Boxplot and Violin Element, and look
at how the plot has changed. Next , tick Boxplot, Violin and Jitter Elements. The Violin plot has taken
the smoothed distribution curve from the Distribution plot, rotated it 90o and superimposed it on the
boxplot.  The jitter plot has further added all the data points.
If your data is split by group , for example, the boxplots for each group will be shown on the same
graph, and the colours of each will be different if the Colour palette is ticked. 5 colour palettes are
available.
Maximum value
Median value
Minimum value
75% quartile
25% quartile
IQR
Top 25%
Bottom 25%
Outlier
Boxplot + Violin plot
Boxplot + Violin + Jitter plot

Scatter Plots
JASP can produce scatterplots of various types and can include smooth or linear regression lines.
There are also options to add distributions to these, either in the form of density plots or
histograms.
GGplot2 palette
Viridis palette

Tile Heatmap
These plots provide an alternative way of visualising data. For example , using the Titanic survival
dataset to look at the relationship between the class of passage and survival.

# CUSTOMISABLE PLOTS (page 28)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_CUSTOMISABLE_PLOTS_PAGE_28]

# EDITING PLOTS (page 32)


Clicking on the drop-down menu provided access to a range of options, including Edit Image.
Selecting this option provides some customisation for each graph.
This will open the plot in a new window, which allows some modifications of each axis in terms of
axis title and range.
Any changes are then updated in the results window. The new plot can be saved as an image or can
be reset to default values.

Do not forget that group labels can be changed in the spreadsheet editor.

# EDITING PLOTS (page 32)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_EDITING_PLOTS_PAGE_32]

# EXPLORING DATA INTEGRITY (page 34)


Sample data is used to estimate parameters of the population, whereby a parameter is a measurable
characteristic of a population, such as a mean, standard deviation, standard error or confidence
intervals, etc.
What is the difference between a statistic and a parameter? If you randomly polled a selection of
students about the quality of their student bar , you may find that 75% of them were happy with it.
That is a sample statistic since only a sample of the population w as asked. You calculated what the
population was likely to do based on the sample. If you asked all the students in the university, and
90% were happy, you have a parameter since you asked the whole university population.
Bias can be defined as  the tendency of a measurement  to over - or underestimate the value of a
population parameter. Many types of bias can appear in research design and data collection, including:
Participant selection bias  some being more likely to be selected for study than others
Participant exclusion bias - due to the systematic exclusion of certain individuals from the
study
Analytical bias - due to the way that the results are evaluated
However, statistical bias can affect a) parameter estimates, b) standard errors and confidence
intervals or c) test statistics and p values.  So, how can we check for bias?

# EXPLORING DATA INTEGRITY (page 34)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_EXPLORING_DATA_INTEGRITY_PAGE_34]

# IS YOUR DATA CORRECT? (page 34)


Outliers are data points that are abnormally outside all other data points. Outliers can be due to a
variety of things, such as errors in data input or analytical errors at the point of data collection Boxplots
are an easy way to visualise such data points where outliers are outside the upper (75% + 1.5 * IQR)
or lower (25% - 1.5 * IQR) quartiles
Boxplots show:
Median value
25 & 75% quartiles
IQR  Inter quartile range
Max & min values plotted
with outliers excluded
Outliers shown if requested

Load Exploring Data.csv into JASP. Under Descriptives > Descriptive Statistics, add Variable 1 to the
Variables box. In Plots, tick the following: Boxplots, Label Outliers, and BoxPlot Element.
The resulting Boxplot on the left looks very compressed, and an obvious outlier is labelled as being in
row 38 of the dataset. This can be traced back to a data input error in which 91.7 was input instead of
917.  The graph on the right shows the BoxPlot for the clean data.

How you deal with an outlier depend s on the cause. Most parametric tests are highly sensitive to
outliers, while non-parametric tests are generally not.
Correct it?  Check the original data to make sure that it isnt an input error; if it is, correct it, and rerun
the analysis.
Keep it? - Even in datasets of normally distributed data , outliers may be expected for large sample
sizes and should not automatically be discarded if that is the case.
Delete it?  This is a controversial practice in small datasets where a normal distribution cannot be
assumed. Outliers resulting from an instrument reading error may be excluded , but they should be
verified first.
Replace it?  Also known as winsorizing. This technique replaces the outlier values with the relevant
maximum and/or minimum values found after excluding the outlier.
Whatever method you use must be justified in your statistical methodology and subsequent analysis.

# IS YOUR DATA CORRECT? (page 34)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_IS_YOUR_DATA_CORRECT_PAGE_34]

# WE MAKE MANY ASSUMPTIONS ABOUT OUR DATA. (page 36)


When using parametric tests, we make a series of assumptions about our data and bias will occur if
these assumptions are violated:
Normality
Homogeneity of variance or homoscedasticity
Many statistical tests are an omnibus of tests, of which some will check these assumptions.

# WE MAKE MANY ASSUMPTIONS ABOUT OUR DATA. (page 36)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_WE_MAKE_MANY_ASSUMPTIONS_ABOUT_OUR_DATA_PAGE_36]

# TESTING THE ASSUMPTION OF NORMALITY (page 36)


Normality does not necessarily mean that the data is normally distributed per se, but it is whether or
not the dataset can be modelled by a normal distribution. Normality can be explored in a variety of
ways:
Numerically
Visually / graphically
Statistically
Numerically, we can use the Descriptives output to calculate skewness and kurtosis. For a normal data
distribution, both values should be close to zero. To determine the significance of skewness or
kurtosis, we calculate their z-scores by dividing them by their associated standard errors:
Skewness Z =
skewness
Skewness standard error          Kurtosis Z =
kurtosis
kurtosis standard error
Z score significance: p<0.05 if z >1.96        p<0.01 if z >2.58      p<0.001 if z >3.29

Using Exploring data.csv, go to Descriptives>Descriptive Statistics move Variable 3 to the Variables
box, in the Statistics drop -down menu select Mean, Std Deviation, Skewness and Kurtosis as shown
below with the corresponding output table.
Neither skewness nor kurtosis is close to 0. The positive skewness suggests that the data is distributed
more on the left (see graphs later) , while the negative kurtosis suggests a flat distribution. When
calculating their z-scores, it can be seen that the data is significantly skewed, p<0.05.
Skewness Z =
0.839
0.337     =   2.49    Kurtosis Z =
-0.407
0.662  = 0.614
[As a note of caution, skewness and kurtosis may appear significant in large datasets even though the
distribution is normal.]
Now add Variable 2 to the Variables box , and in Plots , tick Distribution plot.  This will show the
following two graphs:

It is quite easy to visualise that Variable 2 has a symmetrical distribution. Variable 3 is skewed to the
left, as confirmed by the skewness Z score.
Another graphical check for normality is a Q -Q plot.  Q-Q plots are available in Descriptives and are
also produced as part of the Assumption Checks used in linear regression and ANOVA. Q-Q plots show
the quantiles of the actual data against those expected for a normal distribution.
If the data are normally distributed , all the points will be close to the diagonal reference line. If the
points sag above or below the line , there is a problem with kurtosis. If the points snake around the
line, then the problem is skewness.  Below are Q-Q plots for Variables 2 and 3.  Compare these to the
previous distribution plots and the skewness/kurtosis z scores above.
The following Q-Q plot scenarios are possible:
Variable 2
Variable 3

The Shapiro-Wilk test is a statistical test used by JASP to check the assumption of normality.  It is also
used in the Independent (distribution of the two groups) and Paired (distribution of differences
between pairs) t-tests. The test results in a W value, where small values indicate your sample is not
normally distributed (the null hypothesis that your population is normally distributed , if your values
are under a certain threshold, can, therefore, be rejected).
In Descriptives, the Shapiro -Wilk test can be selected in the Distribution tests.  The Shapiro-Wilk
output table show s no significant deviation in normality for Variable 2 but a significant deviation
(p<.001) for Variable 3.
The most important limitation is that the test can be biased by sample size. The larger the sample, the
more likely youll get a statistically significant result.
Testing the assumption of normality  A cautionary note!
For most parametric tests to be reliable , one of the assumptions is that the data is approximately
normally distributed. A normal distribution peaks in the middle and is symmetrical about the mean.
However, data does not need to be perfectly normally distributed for the tests to be reliable.
So, having gone on about testing for normality, is it necessary?
The Central Limit Theorem states that as the sample size gets larger , i.e. >30 data points , the
distribution of the sampling means approaches a normal distribution.  So the more data points you
have, the more normal the distribution will look and the closer your sample mean approximates the
population mean.
Large datasets may result in significant tests of normality, i.e. Shapiro-Wilk or significant skewness and
kurtosis z -scores when the distribution graphs look fairly normal.  Conversely, small datasets will
reduce the statistical power to detect non-normality.
However, data that does not meet the assumption of normality is going to result in poor results for
certain types of tests (i.e. ones that state that the assumption must be met!). How closely does your
data need to be normally distributed? This is a judgment call best made by eyeballing the data.

# TESTING THE ASSUMPTION OF NORMALITY (page 36)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TESTING_THE_ASSUMPTION_OF_NORMALITY_PAGE_36]

# WHAT DO I DO IF MY DATA IS NOT NORMALLY DISTRIBUTED? (page 40)


Transform the data and redo the normality checks on the transformed data.  Common transformations
include taking the log or square root of the data. Please refer to the next chapter.
Use non -parametric tests since these are distribution -free tests and can be used instead of their
parametric equivalent.

# WHAT DO I DO IF MY DATA IS NOT NORMALLY DISTRIBUTED? (page 40)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_WHAT_DO_I_DO_IF_MY_DATA_IS_NOT_NORMALLY_DISTRIBUTED_PAGE_40]

# TESTING HOMOGENEITY OF VARIANCE (page 40)


Levenes test is commonly used to test the null hypothesis that variances in different groups are equal.
The result from the test (F) is reported as a p-value; if not significant, then you can say that the null
hypothesis stands  that the variances are equal; if the p-value is significant, then the implication is
that the variances are unequal.  Levenes test is included in the Independent t-test and ANOVA in
JASP as part of the Assumption Checks.
Using Exploring data.csv, go to T-Tests>Independent Samples t-test, move Variable 1 to the Variables
box and Group to the Grouping variable and tick Assumption Checks > Equality of variances.

In this case, there is no significant difference  in variance between the two groups , F (1) = 0.218, p
=.643.
The assumption of homoscedasticity ( equal variance) is important in linear regression models, as is
linearity.  It assumes that the variance of the data around the regression line is the same for all
predictor data points.   Heteroscedasticity (the violation of homoscedasticity) is present when the
variance differs across the values of an independent variable.  This can be visually assessed in linear
regression by plotting actual residuals against predicted residuals.
If homoscedasticity and linearity are not violated, there should be no relationship between what the
model predicts and its errors, as shown in the graph on the left. Any sort of funnelling (middle graph)
suggests that homoscedasticity has been violated, and any curve (right graph) suggests that linearity
assumptions have not been met.

# TESTING HOMOGENEITY OF VARIANCE (page 40)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TESTING_HOMOGENEITY_OF_VARIANCE_PAGE_40]

# DATA TRANSFORMATION (page 42)


JASP provides the ability to compute new variables or transform data.  In some cases, it may
be useful to compute the differences between repeated measures , or, to make a dataset
more normally distributed, you can apply a log transform, for example.
When a dataset is opened, there will be a plus sign (+) at the end of the columns.
Clicking on the + opens a small dialogue window where you can;
Enter the name of a new variable or the transformed variable
Select whether you enter the R code directly or use the commands built into JASP
Select what data type is required
Once you have named the new variable and chosen the other options, click create.

If you choose the manual option rather than the R code, this opens all the built-in create and
transform options. Although not obvious, you can scroll the left and right-hand options to see
more variables or more operators, respectively.
For example, we want to create a column of data showing the difference between variable 2
and variable 3. Once you have entered the column name in the Create Computed Column
dialogue window, its name will appear in the spreadsheet window.  The mathematical
operation now needs to be defined.  In this case, drag variable 2 into the equation box, drag
the minus sign down and then drag in variable 3.
If you have made a mistake, i.e. used the wrong variable or operator, remove it by dragging
the item into the dustbin in the bottom right corner.

When you are happy with the equation/operation, click compute column and the data will
be entered.
If you decide that you do not want to keep the derived data, you can remove the column by
clicking the other dustbin icon next to the R.
Another example is to do a log transformation of the data. In the following case , variable 1
has been transformed by scrolling the operators on the left and selecting the log10(y) option.
Replace the y with the variable that you want to transform , and then click Compute
column. When finished, click the X to close the dialogue.
The Export function will also export any new data variables that have been created.

The two graphs below show the untransformed and the log10-transformed data. The skewed
data has been transformed into a profile with a more normal distribution.
Untransformed
Log10 transformed

# DATA TRANSFORMATION (page 42)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_DATA_TRANSFORMATION_PAGE_42]

# EFFECT SIZE (page 46)


When performing a hypothesis test on data , we determine the relevant statistic (r, t, F , etc) and p -
value to decide whether to accept or reject the null hypothesis. A small p -value, <0.05 in most
analyses, provides evidence against the null hypothesis , whereas a large p -value >0.05 only means
that there is not enough evidence to reject the null hypothesis. A lower p -value is sometimes
incorrectly interpreted as meaning there is a stronger relationship or difference between variables. So
what is needed is not just null hypothesis testing but also a method of determining precisely how large
the effects seen in the data are.
An effect size is a statistical measure used to determine the strength of the relationship or difference
between variables.  Unlike a p-value, effect sizes can be used to quantitatively compare the results
of different studies.
For example, comparing heights between 11 and 12-year-old children may show that the 12-year-
olds are significantly taller, but it is difficult to visually see a difference, i.e. small effect size.
However, a significant difference in height between 11 and 16-year-old children is obvious (large
effect size).
The effect size is usually measured in three ways:
The standardised mean difference
correlation coefficient
odds ratio
When looking at differences between groups, most techniques are primarily based on the differences
between the means divided by the average standard deviations. The values derived can then be used
to describe the magnitude of the differences.  The effect sizes calculated in JASP for t-tests and ANOVA
are shown below:

When analysing bivariate or multivariate relationships, the effect sizes are the correlation
coefficients:
When analysing categorical relationships via contingency tables , i.e. chi-square test, Phi is only used
for 2x2 tables while Cramers V and be used for any table size.
For a 2  2 contingency table, we can also define the odds ratio measure of effect size.

# EFFECT SIZE (page 46)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_EFFECT_SIZE_PAGE_46]

# ONE-SAMPLE T-TEST (page 48)


Research is normally carried out in sample populations, but how close ly does the sample reflect the
whole population? The parametric one-sample t -test determines whether the sample mean is
statistically different from a known or hypothesised population mean.
The null hypothesis (Ho) tested is that the sample mean is equal to the population mean.
ASSUMPTIONS
Three assumptions are required for a one-sample t-test to provide a valid result:
The test variable should be measured on a continuous scale.
The test variable data should be independent, i.e. no relationship between any of the data
points.
The data should be approximately normally distributed.
There should be no significant outliers.

# ONE-SAMPLE T-TEST (page 48)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_ONE_SAMPLE_T_TEST_PAGE_48]

# RUNNING THE ONE-SAMPLE T-TEST (page 49)


Open one sample t -test.csv, which contains two columns of data representing the height (cm) and
body mass (kg) of a sample population of males used in a study. In 2017, the average adult male in the
UK population was 178 cm tall and had a body mass of 83.4 kg.
Go to T-Tests > One-Sample t-test and in the first instance, add height to the analysis box on the right.
Then tick the following options above and add 178 as the test value.

# RUNNING THE ONE-SAMPLE T-TEST (page 49)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_ONE_SAMPLE_T_TEST_PAGE_49]

# UNDERSTANDING THE OUTPUT (page 49)


The output should contain three tables and two graphs.
The assumption check of normality (Shapiro -Wilk) is not significant , suggesting that the heights are
normally distributed; therefore, this assumption is not violated. If this showed a significant difference,
the analysis should be repeated using the non -parametric equivalent, Wilcoxons signed -rank test,
tested against the population median height.
This table shows that there are no significant differences between the means, p =.706
The descriptive data shows that the mean height of the sample population was 177.6 cm compared
to the average of 178 cm for UK males.

The two plots show essentially the same data but in different ways. The standard Descriptive plot is
an Interval plot showing the sample mean (black bullet), and the 95% confidence interval (whiskers),
relative to the test value (dashed line).
The Raincloud Plot shows the data as individual data points, a box plot, and the distribution plot.
This can be shown as either a vertical or horizontal display.

Repeat the procedure by replacing height with mass and changing the test value to 83.4.
The assumption check of normality (Shapiro-Wilk) is not significant, suggesting that the masses are
normally distributed.
This table shows that there is a significant difference between the mean sample (72.9 kg) and
population body mass (83.4 kg), p <.001

# UNDERSTANDING THE OUTPUT (page 49)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_49]

# REPORTING THE RESULTS (page 51)


A one-sample t-test showed no significant difference in height compared to the population mean ( t
(22) = -0.382, p .706); however, the participants were significantly lighter than the UK male population
average (t (22) =-7.159, p<.001).

# REPORTING THE RESULTS (page 51)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_51]

# BINOMIAL TEST (page 52)


The binomial test is effectively a non -parametric version of the one -sample t -test for use with
dichotomous (i.e. yes/no) categorical datasets. This tests if the sample frequency is statistically
different from a known or hypothesised population frequency.
The null hypothesis (Ho) tested is that the sample data frequency is equal to the expected population
frequency.
ASSUMPTIONS
Three assumptions are required for a binomial test to provide a valid result:
The test variable should be a dichotomous scale (such as yes/no, male/female, etc.).
The sample responses should be independent
The sample size is smaller, but representative of the population

# BINOMIAL TEST (page 52)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_BINOMIAL_TEST_PAGE_52]

# RUNNING THE BINOMIAL TEST (page 52)


Open binomial.csv; this contains one column of data showing the number of students using either a
Windows laptop or a MacBook at university. In January 2021, when comparing just the two operating
systems, the relative UK market share of Windows was 72% and macOS 28%.3
Go to Frequencies >Binomial test. Move the Laptop variable to the data window and set the Test value
to 0.72 (72%). Also, tick the Descriptive plots.
3https://www.statista.com/statistics/487002/market-share-of-desktop-operating-systems-
uk/#:~:text=Apple%20and%20Microsoft%20own%20a,26.48%20percent%20of%20the%20market.

The following table and graph show that the frequencies of both laptops are significantly less than
72%. In particular, students are using significantly fewer Windows laptops than was expected
compared to the UK market share.
Is this the same for MacBook users? Go back to the Options window and change the test value to
0.28 (14%).  This time, both frequencies are significantly higher than 28%. This shows that students
are using significantly more MacBooks than was expected compared to the UK market share.

# RUNNING THE BINOMIAL TEST (page 52)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_BINOMIAL_TEST_PAGE_52]

# REPORTING THE RESULTS (page 54)


The UK relative proportion of  Windows and MacBook users was reported to be 72% and 28%
respectively. In a cohort of university students (N=90), a Binomial test revealed that the proportion of
students using Windows laptops was significantly less ( 60%, p=.014) and those using MacBooks
significantly more (40%, p=.014) than expected.

# REPORTING THE RESULTS (page 54)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_54]

# MULTINOMIAL TEST (page 55)


The multinomial test is effectively a n extended version of the Binomial test for use with categorical
datasets containing three or more factors . This tests whether or not the sample frequency is
statistically different from a hypothesi sed population frequency (multinomial test) or a known (Chi-
square goodness-of-fit test).
The null hypothesis (H o) tested is that the sample frequency is equal to the expected population
frequency.
ASSUMPTIONS
Three assumptions are required for a multinomial test to provide a valid result:
The test variable should be a categorical scale containing 3 or more factors
The sample responses should be independent
The sample size is smaller, but representative of the population

# MULTINOMIAL TEST (page 55)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_MULTINOMIAL_TEST_PAGE_55]

# RUNNING THE MULTINOMIAL TEST (page 55)


Open multinomial.csv. This contains three columns of data showing the number of different coloured
M&Ms counted in five bags.  Without any prior knowledge, it could be assumed that the different
coloured M&Ms are equally distributed.
Go to Frequencies > Multinomial test. Move the colour of the M&Ms to Factor and the observed
number of M&Ms to counts. Tick Descriptives and Descriptives Plots.

As can be seen in the Descriptive table, the test assumes an equal expectation for the proportions of
coloured M&Ms (36 of each colour).  The Multinomial test results show that the observed distribution
is significantly different (p<.001) from an equal distribution.

# RUNNING THE MULTINOMIAL TEST (page 55)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_MULTINOMIAL_TEST_PAGE_55]

# REPORTING THE RESULTS (page 56)


A multinomial test showed that there was a significant difference between the observed
and expected M&M colour distribution: 2 (5) = 35.9, p<.001.

# REPORTING THE RESULTS (page 56)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_56]

# CHI-SQUARE ‘GOODNESS-OF-FIT’ TEST. (page 57)


However, further research shows that the manufacturer produces coloured M&Ms in different ratios:
Colour Blue Brown Green Orange Red Yellow
Proportion 24 13 16 20 13 14
These values can now be used as the expected counts, so move the Expected variable to the Expected
Counts box.  This automatically runs the 2 goodness-of-fit test, leaving the Hypothesis options
greyed out.
As can be seen in the Descriptives table, JASP has calculated the expected numbers of the different
coloured M&Ms based on the manufacturers' reported production ratio.   The results of the test show
that the observed proportions of the different coloured M&Ms are significantly different ( 2 =74.5,
p<.001) from those proportions stated by the manufacturer.

# CHI-SQUARE ‘GOODNESS-OF-FIT’ TEST. (page 57)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_CHI_SQUARE_GOODNESS_OF_FIT_TEST_PAGE_57]

# MULTINOMIAL AND Χ2 ‘GOODNESS-OF-FIT’ TEST. (page 58)


JASP also provides another option whereby bot h tests can be run at the same time. Go back to the
Options window and only add Colour to the Factor and Observed to the Counts boxes , and remove
the expected counts if the variable is still there . In Hypotheses, now tick the 2 test. This will open a
small spreadsheet window showing the colour and Ho (a) with each cell having 1 in it. This is assuming
that the proportions of each colour are equal (multinomial test).
In this window, add another column which will automatically be labelled H o (b). The expected
proportions of each colour can now be typed in.
Now, when the analysis is run , the results of the tests for the two hypotheses are shown. H o (a) is
testing the null hypothesis that the proportions of each colour are equally distributed, while H o (b) is
testing the null hypothesis that the proportions are the same as those expected. As can be seen, both
hypotheses are rejected.  Evidence indicates that the colo urs of M&M's do not match the
manufacturer's published proportions.

Reporting the results
A 2 goodness of fit test was carried out to assess the distributions of M& Ms in retail packets. This
showed that firstly, the proportions of each colour are not equally distributed, 2 (5) = 35.9, p<.001.
Secondly, the distribution of colours w as significantly different to the manufacturer 's stated
distribution, 2 (5) = 74.5, p<.001.

# MULTINOMIAL AND Χ2 ‘GOODNESS-OF-FIT’ TEST. (page 58)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_MULTINOMIAL_AND_2_GOODNESS_OF_FIT_TEST_PAGE_58]

# COMPARING TWO INDEPENDENT GROUPS: INDEPENDENT T-TEST (page 60)


The parametric independent t-test, also known as Students t -test, is used to determine if there is a
statistical difference between the means of two independent groups. The test requires a continuous
dependent variable (i.e. body mass) and an independent variable comprising 2 groups (i.e. males and
females).
This test produces a t -score, which is a ratio of the differences between the two groups and the
differences within the two groups:
$$t = \frac{\text{mean group 1 - mean group 2}}{\text{standard error of mean differences}}$$
mean group 1  mean group 2
standard error of the mean differences
A large t -score indicates that there is a greater difference between groups. The smaller the t -score,
the more similarity there is between groups. A t-score of 5 means that the groups are five times as
different from each other as they are within each other.
The null hypothesis (H o) tested is that the population means from the two unrelated groups are
equal.

# COMPARING TWO INDEPENDENT GROUPS: INDEPENDENT T-TEST (page 60)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_COMPARING_TWO_INDEPENDENT_GROUPS_INDEPENDENT_T_TEST_PAGE_60]

# ASSUMPTIONS OF THE PARAMETRIC INDEPENDENT T-TEST (page 60)


Group independence:
Both groups must be independent of each other. Each participant will only provide one data point for
one group. For example, participant 1 can only be in either a male or female group, not both. Repeated
measures are assessed using the Paired t-test.
Normality of the dependent variable:
The dependent variable should also be measured on a continuous scale and be approximately
normally distributed with no significant outliers. This can be checked using the Shapiro-Wilk test. The
t-test is fairly robust , and small deviations from normality are normally acceptable. However, this is
not the case if the group sizes are very different. A rule of thumb is that the ratio between the group
sizes should be <1.5 (i.e. group A = 12 participants and group B = >8 participants).
If normality is violated, you can try transforming your data (for example, log values, square root values)
or, if the group sizes are very different, use the Mann-Whitney U test, which is a non -parametric
equivalent that does not require the assumption of normality (see later).
X = mean
S = standard deviation
n = number of data points

Homogeneity of variance:
The variances of the dependent variable should be equal in each group. This can be tested using
Levene's Test of Equality of Variances.
If Levene's test is statistically significant, indicating that the group variances are unequal , we can
correct for this violation by using an adjusted t-statistic based on the Welch method.

# ASSUMPTIONS OF THE PARAMETRIC INDEPENDENT T-TEST (page 60)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_ASSUMPTIONS_OF_THE_PARAMETRIC_INDEPENDENT_T_TEST_PAGE_60]

# RUNNING THE INDEPENDENT T-TEST (page 61)


Open Independent t-test.csv, which contains weight loss on a self -controlled 10-week diet between
men and women. It's good practice to check the Distribution and boxplots in Descriptives to visually
check for distribution and outliers.
Go to T -Tests > Independent Samples t -test and put weight loss in the Dependent variable box and
gender (independent variable) in the Grouping Variable box.
Unequal variance
Equal variance

In the analysis window, tick the following options:

# RUNNING THE INDEPENDENT T-TEST (page 61)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_INDEPENDENT_T_TEST_PAGE_61]

# UNDERSTANDING THE OUTPUT (page 62)


The output should consist of four tables and three graphs. Firstly, we need to check that the
parametric assumptions required are not violated.
The Shapiro-Wilk test shows that both groups have normally distributed data ; therefore, the
assumption of normality is not violated.  If one or both were significant, you should consider using the
non-parametric equivalent Mann-Whitney test.

Levenes test shows that there is no difference in the variance; therefore, the assumption of
homogeneity of variance is not violated.  If Levenes test was significant, Welchs adjusted t-statistic,
degrees of freedom and p-values should be reported.
This table shows the two computed t -statistics (Student and Welch ). Remember, the t -statistic is
derived from the mean difference divided by the standard error of the difference. Both show that
there is a significant statistical difference between the two groups ( p<.001), and Cohens d suggests
that this is a large effect.

From the descriptive data and plots (use whichever you want), it can be seen that females had a higher
weight loss than males.

# UNDERSTANDING THE OUTPUT (page 62)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_62]

# REPORTING THE RESULTS (page 64)


A Levene test found that the assumption of homogeneity of variance was met:  F(1) = 2.28, p=.135. A
Shapiro-Wilk test shows that both groups have normally distributed data: Females ( W=0.97, p=.282)
and m ales ( W=0.97, p=0.31). Therefore, an independent sample  Student t -test based on equal
variances was carried out. This showed that females lost significantly more weight  (M = 6.93, SD =
2.24) over 10 weeks of dieting than males (M =3.72, SD = 2.56):  t(85)=6.16, p<.001.  Cohens d (1.322)
suggests that this is a large effect.
NB if the Levene test is significant, the Welch t-statistic and corrected df should be reported.

# REPORTING THE RESULTS (page 64)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_64]

# MANN-WITNEY U TEST (page 65)


If you find that your data is not normally distributed (such as a significant Shapiro-Wilk test result) or
is ordinal by nature, the equivalent non-parametric independent test is the Mann-Whitney U test.
Open Mann-Whitney pain.csv , which contains subjective pain scores (0 -10) with and without ice
therapy. NOTE: Make sure that Treatment is categorical, and the pain score is ordinal. Go to T-Tests >
Independent t-tests, put pain score in the Dependent variable box and use Treatment as the grouping
variable.
In the analysis options, only tick:
Mann-Whitney
Location parameter
Effect size
There is no reason to repeat the assumption checks since Mann -Whitney does not require the
assumption of normality or homogeneity of variance required by parametric tests.

# MANN-WITNEY U TEST (page 65)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_MANN_WITNEY_U_TEST_PAGE_65]

# UNDERSTANDING THE OUTPUT (page 65)


This time, you will only get one table:
The Mann-Whitney U-statistic is highly significant.  W=207, p<.001.
The location parameter, the Hodges Lehmann estimate, is a measure of the  median
difference between the two groups. The rank-biserial correlation (rB) can be considered as an effect
size and is interpreted the same as Pearsons r, so 0.84 is a large effect size.
For non -parametric data, you should report median values as your descriptive statistics and use
boxplots instead of line graphs and confidence intervals, SD/SE bars. Go to Descriptive statistics, put
the Pain score into the variable box and split the file by Treatment.
You can alternatively show the Raincloud plots for more information on the differences and
distributions.

# UNDERSTANDING THE OUTPUT (page 65)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_65]

# REPORTING THE RESULTS (page 66)


A Mann-Whitney test showed that Ice therapy significantly reduces pain scores ( Mdn = 3) compared
to the control group (Mdn = 7): W=207, p<.001.  The rB= 0.84 suggests that this is a large effect size.

# REPORTING THE RESULTS (page 66)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_66]

# COMPARING TWO RELATED GROUPS: PAIRED SAMPLES T-TEST (page 67)


As with the Independent t-test, there are both parametric and non -parametric options available in
JASP. The parametric paired-samples t-test (also known as the dependent sample t -test or repeated
measures t -test) compares the means between two related groups on the same continuous,
dependent variable. For example, looking at weight loss pre- and post-10 weeks of dieting.
The paired t-statistic =
mean of the differences between group pairs
the standard error of the mean differences
With the paired t-test, the null hypothesis (H o) is that the pairwise difference between the two
groups is zero.

# COMPARING TWO RELATED GROUPS: PAIRED SAMPLES T-TEST (page 67)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_COMPARING_TWO_RELATED_GROUPS_PAIRED_SAMPLES_T_TEST_PAGE_67]

# ASSUMPTIONS OF THE PARAMETRIC PAIRED SAMPLES T-TEST (page 67)


Four assumptions are required for a paired t-test to provide a valid result:
The dependent variable should be measured on a continuous scale.
The independent variable should consist of 2 categorical related/matched groups, i.e. each
participant is matched in both groups.
The differences between the matched pairs should be approximately normally distributed.
There should be no significant outliers in the differences between the 2 groups.

# ASSUMPTIONS OF THE PARAMETRIC PAIRED SAMPLES T-TEST (page 67)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_ASSUMPTIONS_OF_THE_PARAMETRIC_PAIRED_SAMPLES_T_TEST_PAGE_67]

# RUNNING THE PAIRED SAMPLES T-TEST (page 67)


Open Paired t-test.csv in JASP. This contains two columns of paired data, pre-diet body mass and post-
weeks of dieting. Go to T-Tests > Paired Samples t-test. Ctrl-click both variables and add them to the
analysis box on the right.

In the analysis options, tick the following:

# RUNNING THE PAIRED SAMPLES T-TEST (page 67)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_PAIRED_SAMPLES_T_TEST_PAGE_67]

# UNDERSTANDING THE OUTPUT (page 68)


The output should consist of three tables and three graphs.
The assumption check of normality (Shapiro -Wilk) is not significant , suggesting that the pairwise
differences are normally distributed ; therefore, the assumption is not violated. If this showed a
significant difference , the analysis should be repeated using the non -parametric equivalent ,
Wilcoxons signed-rank test.

This shows that there is a significant difference in body mass between the pre - and post -dieting
conditions, with a mean difference (location parameter) of 3.78kg.  Cohens d states that this is a large
effect.
The descriptive statistics and plot show that there was a reduction in body mass following 4 weeks of
dieting.
The Raincloud plot show s the data for both groups and the distribution of the differences between
the two groups.

The Q-Q plot reinforces the fact that the data is normally distributed; the closer to the diagonal line
the points lie, the more normally distributed the data.

# UNDERSTANDING THE OUTPUT (page 68)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_68]

# REPORTING THE RESULTS (page 70)


The Shapiro-Wilk test was not significant, suggesting that the pairwise differences were normally
distributed; therefore, the assumption is not violated: W=0.98, p=.124.
Post-diet body mass was lower (M = 68.7, SD =9.0 kg) than at the start of the diet (M =725, SD = 8.7kg).
This mean difference ( M = 3.78, SE = 0.29 kg) was shown to be significant: t (77) =13.04, p<.001.
Cohens d (1.48) suggests that this is a large effect.

# REPORTING THE RESULTS (page 70)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_70]

# RUNNING THE NON-PARAMETRIC PAIRED SAMPLES TEST: WILCOXON’S SIGNED RANK TEST (page 71)


If you find that your data is not normally distributed (significant Shapiro-Wilk test result) or is ordinal
by nature, the equivalent non -parametric independent test is Wilcoxons signed -rank test .  Open
Wilcoxons rank.csv . This has two columns , one with pre -anxiety and post-hypnotherapy anxiety
scores (from 0 - 50).  In the dataset view, make sure that both variables are assigned to the ordinal
data type.
Go to T-Tests > Paired Samples t-test and follow the same instructions as above, but now only tick the
following options:
There will be only one table in the output:
The Wilcoxon W-statistic is highly significant, p<0.001.
The location parameter, the Hodges Lehmann estimate, is the median difference between the two
groups. The rank -biserial correlation (r B) can be considered as an effect size and is interpreted the
same as Pearsons r, so 0.48 is a medium to large effect size.
Effect size Trivial Small Medium Large
Rank -biserial (rB) <0.1
0.1
0.3
0.5

For non -parametric data, you should report median values as your descriptive statistics and use
boxplots or Rainbow plots instead of line graphs and confidence intervals, SD/SE bars.

# RUNNING THE NON-PARAMETRIC PAIRED SAMPLES TEST: WILCOXON’S SIGNED RANK TEST (page 71)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_NON_PARAMETRIC_PAIRED_SAMPLES_TEST_WILCOXON_S_SIGNED_RANK_TEST_PAGE_71]

# REPORTING THE RESULTS (page 72)


A Wilcoxon signed-rank test showed that hypnotherapy significantly reduces anxiety scores ( Mdn =
15) compared to pre -therapy (Mdn =22) scores, W=322, p<.001. The rank-biserial correlation (r B) =
0.982 suggests that this is a large effect size.

# REPORTING THE RESULTS (page 72)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_72]

# CORRELATION ANALYSIS (page 73)


Correlation is a statistical technique that can be used to determine if, and how strongly, pairs of
variables are associated.  Correlation is only appropriate for quantifiable data in which numbers are
meaningful, such as continuous or ordinal data. It cannot be used for purely categorical data, for which
we have to use contingency table analysis (see Chi-square analysis in JASP).
Essentially, do different variables co-vary? i.e. are changes in one variable reflected in similar changes
to another variable?  If one variable deviates from its mean , does the other variable deviate from its
mean in either the same or opposite direction?  This can be assessed by measuring covariance ;
however, this is not standardised. For example, we can measure the covariance of two variables that
are measured in meters ; however, if we convert the same values to centimetres, we get the same
relationship but with a completely different covariance value.
To overcome this, standardised covariance is used , which is known as Pearsons correlation
coefficient (or "r"). It ranges from -1.0 to +1.0. The closer r is to +1 or -1, the more closely the two
variables are related. If r is close to 0, there is no relationship. If r is (+), then as one variable increases,
the other also increases. If r is (-), then as one increases, the other decreases (sometimes referred to
as an "inverse" correlation).
The correlation coefficient (r) should not be confused with R 2 (coefficient of determination) or R
(multiple correlation coefficient as used in the regression analysis).
The main assumption in this analysis is that the data have a normal distribution and are linear. This
analysis will not work well with curvilinear relationships.
Covariance = 4.7 Covariance = 470

# CORRELATION ANALYSIS (page 73)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_CORRELATION_ANALYSIS_PAGE_73]

# RUNNING CORRELATION (page 74)


The analysis tests the null hypothesis (H0) that there is no association between the two variables.
From the example data, open Jump height correlation.csv, which contains 2 columns of data, jump
height (m) and explosive leg power (W). Firstly, run the Descriptive statistics and check the box plots
for any outliers.
To run the correlation analysis , go to Regression > Correlation.  Move the 2 variables to the analysis
box on the right. Tick
Pearson,
Report significance,
Flag significant correlations
Under Plots
Scatter plots
Heatmap

# RUNNING CORRELATION (page 74)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_CORRELATION_PAGE_74]

# UNDERSTANDING THE OUTPUT (page 75)


The first table shows the correlation matrix with Pearsons r-value and its p-value. This shows a highly
significant correlation ( p<.001) with a large r value close to 1 ( r= 0.984), and we can reject the null
hypothesis.
For simple correlations like this, it is easier to look at the pairwise table (go back to analysis and tick
the Display pairwise table option.  This replaces the correlation matrix in the results , which may be
easier to read.
Pearsons r value is an effect size where <0.1 is trivial, 0.1 -0.3 is a small effect, 0.3  0.5 is a moderate
effect, and >0.5 is a large effect.
The plot provides a simple visualisation of this strong positive correlation (r = 0.984, p<.001), which is
also highlighted by the heatmap (more relevant when looking at multiple correlations).

# UNDERSTANDING THE OUTPUT (page 75)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_75]

# GOING ONE STEP FURTHER. (page 76)


If you take the correlation coefficient r and square it , you get the coefficient of determination (R 2).
This is a statistical measure of the proportion of variance in one variable that is explained by the other
variable. Or:
$$R^2 = \frac{SS_M}{SS_T}$$
R2 is always between 0 and 100% where:
0% indicates that the model explains none of the variability of the response data around its
mean, and
100% indicates that the model explains all the variability of the response data around its
mean.
In the example above, r = 0.984, so R 2 = 0.968. This suggests that jump height accounts for 96.8% of
the variance in explosive leg power.

# GOING ONE STEP FURTHER. (page 76)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_GOING_ONE_STEP_FURTHER_PAGE_76]

# REPORTING THE RESULTS (page 76)


Pearsons correlation showed a significant correlation between jump height and leg power (r = 0.984,
p<.001), with jump height accounting for 96.8% of the variance in leg power.

RUNNING NON-PARAMETRIC CORRELATION  Spearmans and Kendalls tau
If your data is ordinal or is continuous data that has violated the assumptions required for parametric
testing (normality and/or variance) , you need to use the non -parametric alternatives to Pearsons
correlation coefficient.
The alternatives are Spearmans (rho) or Kendalls (tau) correlation coefficients. Both are based on
ranking data and are not affected by outliers or normality/variance violations.
Spearman's rho is usually used for ordinal scale data , and Kendall's tau is used in small samples or
when many values with the same score (ties). In most cases, Kendalls tau and Spearmans rank
correlation coefficients are very similar and thus invariably lead to the same inferences.
The effect sizes are the same as Pearsons r. The main difference is that rho 2 can be used as an
approximate non-parametric coefficient of determination, but the same is not true for Kendalls tau.
From the example data, open Non-parametric correlation.csv, which contains 2 columns of data, a
creativity score and position in the Worlds biggest liar competition (thanks to Andy Field).
Run the analysis as before , but now using Spearman and Kendalls tau -b coefficients instead of
Pearsons.

# REPORTING THE RESULTS (page 76)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_76]

# REPORTING THE RESULTS (page 77)


The relationship between creativity scores and final position in the Worlds biggest liar competition
was assessed. A Spearmans correlation test showed a significant relationship: rho = -0.373, p=.002.

# REPORTING THE RESULTS (page 77)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_77]

# NOTE OF CAUTION. (page 78)


The correlation only gives information on the strength of association. It gives no information on the
direction, i.e. which variable causes the other to change. So, it cannot be used to state that one thing
causes the other.  Often , a significant correlation means absolutely nothing and is purely by chance ,
especially if you correlate thousands of variables. This can be seen in the following strange
correlations:
Pedestrians killed in a collision with a railway train correlate with rainfall in Missouri:
The number of honey-producing bee colonies (1000s) correlates strongly with the marriage rate in
South Carolina (per 1000 marriages)

REGRESSION
Whereas correlation tests for associations between variables, regression is the next step commonly
used for predictive analysis, i.e. to predict a dependent outcome variable from one (simple regression)
or more (multiple regression) independent predictor variables.
Regression results in a hypothetical model of the relationship between the outcome and predictor
variable(s).  The model used is a linear one defined by the formula;
y = c + b*x +
y = estimated dependent outcome variable score,
c = constant,
b = regression coefficient and
x = score on the independent predictor variable
= random error component (based on residuals)
Linear regression provides both the constant and regression coefficient(s).
Linear regression makes the following assumptions:
Data (1 dependent variable of continuous level, independent variables that are measured
either at the continuous or nominal level, sample size, etc.)
Outliers, there should be no significant outliers, high-leverage points or highly influential
points.
Linear relationship, there needs to be a linear relationship between (a) the dependent
variable and each of your independent variables, and (b) the dependent variable and the
independent variables collectively.
Independence, there should be independence of observations (i.e., independence of
residuals)
Normality, a) check that the residuals (errors) are approximately normally distributed; and
b) check that the dependent variable is approximately normally distributed.
Equal error variances, data needs to show homoscedasticity of residuals (equal error
variances)
Multicollinearity,  data must show minimal multicollinearity. This occurs when you have
two or more independent variables that are highly correlated with each other.
Concerning sample sizes, there are many different rules of thumb in the literature, ranging from 10-
15 data points per predictor in the model, i.e. 4 predictor variables will each require between 40 and
60 data points each, to 50 +(8 * number of predictors) for each variable. So, for 4 variables, that would
require 82 data point s for each variable. Effectively , the bigger your sample size , the better your
model.

SUMS OF SQUARES (Boring, but the basis of evaluating the regression model.)
Most regression analyses will produce the best model available, but how good is it actually, and how
much error is in the model?
This can be determined by looking at the goodness of fit using the sums of squares. This is a measure
of how close the actual data points are to the modelled regression line.
The vertical difference between the data points and the predicted regression line is known as the
residuals. These values are squared to remove the negative numbers and then summed to give SSR.
This is effectively the error of the model or the goodness of fit, obviously, the smaller the value, the
less error in the model.
Values above the
line are positive
Values below the
line are negative

The vertical difference between the data points and the mean of the outcome variable can be
calculated. These values are squared to remove the negative numbers and then summed to give the
total sum of the squares , SST. This shows how good the mean value is as a model of the outcome
scores.
The vertical difference between the mean of the outcome variable and the predicted regression line
is now determine d. Again, these values are squared to remove the negative numbers and then
summed to give the model sum of squares (SSM). This indicates how better the model is compared to
just using the mean of the outcome variable. SST is the total sum of the squares.
So, the larger the SSM, the better the model is at predicting the outcome compared to the mean value
alone. If this is accompanied by a small SSR, the model also has a small error.
R2 is similar to the coefficient of determination in correlation in that it shows how much of the
variation in the outcome variable can be predicted by the predictor variable(s).
$$R^2 = \frac{SS_M}{SS_T}$$
SST
In regression, the model is assessed by the F statistic based on the improvement in the prediction of
the model SSM and the residual error SSR.  The larger the F value, the better the model.
$$F = \frac{Mean\ SS_M}{Mean\ SS_R}$$
Mean SSR

# NOTE OF CAUTION. (page 78)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_NOTE_OF_CAUTION_PAGE_78]

# SIMPLE REGRESSION (page 82)


Regression tests the null hypothesis (Ho) that there will be no significant prediction of the dependent
(outcome) variable by the predictor variable(s).
Open Rugby kick regression.csv .  This dataset contains rugby kick data , including distance kicked,
right/left leg strength and flexibility and bilateral leg strength.  Firstly, go to Descriptives > Descriptive
statistics and check the boxplots for any outliers. In this case, there should be none, though it is good
practice to check.
For this simple regression, go to Regression > Linear regression and put distance into the Dependent
Variable (outcome) and R_Strength into the Covariates (Predictor) box. Tick the following options in
the Statistics options:

# SIMPLE REGRESSION (page 82)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_SIMPLE_REGRESSION_PAGE_82]

# UNDERSTANDING THE OUTPUT (page 82)


You will now get the following outputs:
Here it can be seen that the correlation (R) between the two variables is high (0.784). , while that for
the null model M0 is 0.  The R2 value of 0.614 tells us that right leg strength accounts for 61.4% of the
variance in kick distance.  Durbin -Watson checks for correlations between residuals, which can
invalidate the test. This should be above 1 and below 3, and ideally around 2.

The ANOVA table shows all the sums of squares mentioned earlier. With regression being the model
and Residual being the error.  The F -statistic is significant - p=.002.  This tells us that the model is a
significantly better predictor of kicking distance than the mean distance.
Report as F (1, 11) = 17.53, p=.002.
This table gives the coefficients (unstandardized) that can be put into the linear equation.
y = c + b*x
y = estimated dependent outcome variable score,
c = constant (intercept)
b = regression coefficient (R_strength)
x = score on the independent predictor variable
For example, for a leg strength of 60 kg, the distance kicked can be predicted by the following:
Distance = 57.105 + (6.452 * 60) = 454.6 m

# UNDERSTANDING THE OUTPUT (page 82)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_82]

# FURTHER ASSUMPTION CHECKS (page 83)


In Plots checks, tick the following two options:

This will result in two graphs:
This graph shows a balanced random distribution of the residuals around the baseline, suggesting that
the assumption of homoscedasticity has not been violated. (See Exploring data integrity in JASP for
further details.
The Q-Q plot shows that the standardised residuals fit nicely along the diagonal, suggesting that both
assumptions of normality and linearity have also not been violated.

# FURTHER ASSUMPTION CHECKS (page 83)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_FURTHER_ASSUMPTION_CHECKS_PAGE_83]

# REPORTING THE RESULTS (page 84)


A simple linear regression was used to predict rugby kicking distance from right leg strength. Leg
strength was shown to explain a significant amount of the variance in the kicking distance: F(1, 11) =
17.53, p.002, R2 = 0.614. The regression coefficient (b = 6.452) allows the kicking distance to be
predicted using the following regression equation:
Distance = 57.105 + (6.452 * Right leg strength)

# REPORTING THE RESULTS (page 84)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_84]

# MULTIPLE REGRESSION (page 85)


The model used is still a linear one defined by the formula.
y = c + b*x +
y = estimated dependent outcome variable score,
c = constant,
b = regression coefficient and
x = score on the independent predictor variable
= random error component (based on residuals)
However, we now have more than 1 regression coefficient and predictor score, i.e.
y = c + b1*x1 + b2*x2 + b3*x3 ........ bn*xn
Data entry methods.
If predictors are uncorrelated , their order of entry has little effect on the model.  In most cases,
predictor variables are correlated to some extent , and thus, the order in which the predictors are
entered can make a difference.  The different methods are subject to much debate in the area.
Enter: This is the default method in which all the predictors are forced into the model in the order
they appear in the Covariates box. This is considered to be the best method.
Blockwise entry (Hierarchical entry): The researcher, normally based on prior knowledge and previous
studies, decides the order in which  the known predictors are entered first , depending on their
importance in predicting the outcome. Additional predictors are added in further steps.
Backwards: All predictors are initially entered in to the model, and then the contribution of each is
calculated.  Predictors with less than a given level of contribution ( p<0.1) are removed. This process
repeats until all the predictors are statistically significant.
Forward: The predictor with the highest simple correlation with the outcome variable is entered first.
Subsequent predictors are selected based on  the size of their semi -partial correlation with the
outcome variable. This is repeated until all predictors that contribute significant unique variance to
the model have been included in the model.
Stepwise entry: Same as the Forward method, except that every time a predictor is added to the
model, a removal test is made of the least useful predictor. The model is constantly reassessed to see
whether any redundant predictors can be removed.
There are many reported disadvantages of using stepwise data entry methods ; however, backwards
entry methods can be useful for exploring previously unused predictors or for fine-tuning the model
to select the best predictors from the available options.

# MULTIPLE REGRESSION (page 85)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_MULTIPLE_REGRESSION_PAGE_85]

# RUNNING MULTIPLE REGRESSION (page 86)


Open Rugby kick regression.csv  that we used for simple regression. Go to Regression > Linear
regression and put distance into the Dependent Variable (outcome) , and now add all the other
variables into the Covariates (Predictor) box.
In the Variable section, leave the Method as Enter. Tick the following options in the Statistics options,
Durbin-Watson, Model fit, Estimates, statistics and Casewise diagnostics.

# RUNNING MULTIPLE REGRESSION (page 86)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_MULTIPLE_REGRESSION_PAGE_86]

# UNDERSTANDING THE OUTPUT (page 87)


You will now get the following outputs:
This provides information on a model based on the M0  (no predictors) and the alternative H1.
The adjusted R2 (used for multiple predictors) shows that they can predict 68.1% of the outcome
variance. Durbin-Watson checks for correlations between residuals are between 1 and 3 as required.
The ANOVA table shows the F -statistic to be significant at p=0.017, suggesting that the model is a
significantly better predictor of kicking distance than the mean distance.
This table shows both the M0 and M1 models and the constant (intercept) and regression coefficients
(unstandardized) for all the predictors forced into the model. Even though the ANOVA shows the
model to be significant, none of the predictor regression coefficients is significant!

The casewise diagnostics table is empty! This is good news. This will highlight any  cases (rows) that
have residuals that are 3 or more standard deviations away from the mean. These cases with the
largest errors may well be outliers. Too many outliers will have an impact on the model and should be
dealt with in the usual way (see Exploring Data Integrity).
As a comparison, re-run the analyses but now choose Backwards as the method of data entry.
The outputs are as follows:
JASP has now calculated 4 potential regression models.  It can be seen that each consecutive model
increases the adjusted R2, with model 4 accounting for 73.5% of the outcome variance.
The Durbin-Watson score is also higher than the forced entry method.
The ANOVA table indicates that each successive model is better , as shown by the increasing F -value
and improving p-value.

Model 1 is the same as the forced entry method first used.  The table shows that as the least
significantly contributing predictors are sequentially removed, we end up with a model with two
significant predictor regression coefficients, right leg strength and bilateral leg strength.
We can now report the Backwards predictor entry results in a highly significant model F(2, 10) = 17.92,
p<.001 and a regression equation of
Distance = 46.251 + (3.914 * R_Strength) + (2.009 * Bilateral Strength)

# UNDERSTANDING THE OUTPUT (page 87)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_87]

# TESTING FURTHER ASSUMPTIONS. (page 89)


As for the simple linear regression example, tick the following options.

The balanced distribution of the residuals around the baseline suggests that the assumption of
homoscedasticity has not been violated.
The Q-Q plot shows that the standardised residuals fit along the diagonal, suggesting that both
assumptions of normality and linearity have also not been violated.

# TESTING FURTHER ASSUMPTIONS. (page 89)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TESTING_FURTHER_ASSUMPTIONS_PAGE_89]

# IN SUMMARY (page 91)


R2 provides information on how much variance is explained by the model using the predictors
provided.
F-statistics provide information as to how good the model is.
The unstandardized (b) -value provides a constant which reflects the strength of the relationship
between the predictor(s) and the outcome variable.
Violation of assumptions can be checked using the Durbin-Watson value, tolerance/VIF values,
Residual vs predicted and Q-Q plots.

# IN SUMMARY (page 91)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_IN_SUMMARY_PAGE_91]

# REPORTING THE RESULTS (page 91)


Multiple linear regression using backwards data entry shows that right leg and bilateral strength can
significantly explain 78.2% of the variance in kicking distance, F(2,10) = 17.92, p<.001, R2 = 0.782. Two
significant regression coefficients were found, right leg strength (b1= 3.91, t=2.59, p=.027) and bilateral
strength (b2 = 2.009, t =2.77, p=0.02). Kicking distance could be predicted using a regression equation
of;
Distance = 57.105 + (3.914 * R_Strength) + (2.009 * Bilateral Strength)

# REPORTING THE RESULTS (page 91)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_91]

# LOGISTIC REGRESSION (page 92)


In simple and multiple linear regression outcome and predictor variable(s) were continuous data.
What if the outcome was a binary/categorical measure?  Can, for example, a yes or no outcome be
predicted by other categorical or continuous variables?  The answer is yes if binary logistic regression
is used. This method is used to predict the probability of a binary yes or no outcome.
The null hypothesis tested is that there is no relationship between the outcome and the predictor
variable(s).
As can be seen in the graph below, a linear regression line between the yes and no responses would
be meaningless as a prediction model. Instead, a sigmoidal logistic regression curve is fitted with a
minimum of 0 and a maximum of 1. Some predictor values overlap between yes and no. For example,
a prediction value of 5 would give an equal 50% probability of being a yes or no outcome. Thresholds
are therefore calculated to determine if a predictor data value will be classified as a yes or no outcome.

# LOGISTIC REGRESSION (page 92)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_LOGISTIC_REGRESSION_PAGE_92]

# ASSUMPTIONS FOR BINARY LOGISTIC REGRESSION (page 92)


The dependent variable must be binary, i.e. yes or no, male or female, good or bad.
One or more independent (predictor variables) can be continuous or categorical variables.
A linear relationship exists between any continuous independent variables and the logit
transformation (natural log of the odds that the outcome equals one of the categories) of the
dependent variable.

# ASSUMPTIONS FOR BINARY LOGISTIC REGRESSION (page 92)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_ASSUMPTIONS_FOR_BINARY_LOGISTIC_REGRESSION_PAGE_92]

# LOGISTIC REGRESSION METRICS (page 92)


AIC (Akaike Information Criteria) and BIC (Bayesian Information Criteria) are measures of fit for the
model; the best model will have the lowest AIC and BIC values.
Outcome = No
Outcome = Yes

Four pseudo R2 values are calculated in JASP: McFadden, Nagelkerke, Tjur and Cox & Snell. These are
analogous to R 2 in linear regression and all give different values .  What constitutes a good R 2 value
varies; however, they are useful when comparing different models for the same data. The model with
the largest R2 statistic is considered to be the best.
The Wald test is used to determine the statistical significance of each of the independent variables.
The confusion matrix is a table showing actual vs predicted outcomes and can be used to determine
the accuracy of the model. From this, sensitivity and specificity can be derived.
Sensitivity is the percentage of cases in which the observed outcome was correctly predicted by the
model (i.e., true positives).
Specificity is the percentage of observations that were also correctly predicted as not having the
observed outcome (i.e., true negatives).

# LOGISTIC REGRESSION METRICS (page 92)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_LOGISTIC_REGRESSION_METRICS_PAGE_92]

# RUNNING LOGISTIC REGRESSION (page 93)


Open Heart attack.csv in JASP. This contains 4 columns of data : Patient ID, did they have a second
heart attack (yes/no), whether they were prescribed exercise (yes/no) and their stress levels (high
value = high stress).
Put the outcome variable (2nd heart attack) into the Dependent variable, add the stress levels to the
Covariates and Exercise prescription to Factors. Leave the data entry method as Enter.
In the Statistics options, tick Estimates, Odds ratios, Confusion matrix, Sensitivity and Specificity.

# RUNNING LOGISTIC REGRESSION (page 93)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_LOGISTIC_REGRESSION_PAGE_93]

# UNDERSTANDING THE OUTPUT (page 94)


The initial output should comprise 4 tables.
The model summary shows that H1 (with the lowest AIC and BIC scores) suggests a significant
relationship ( X2(37) = 21.257, p<.001) between the outcome (2 nd heart attack) and the predictor
variables (exercise prescription and stress levels).
McFadden's R2 = 0.383. It is suggested that a range from 0.2 to 0.4 indicates a good model fit.

Both stress level and exercise prescription are significant predictor variables (p=.031 and .022,
respectively). The most important values in the coefficients table are the odds ratios. For the
continuous predictor, an odds ratio of greater than 1 suggests a positive relationship, while < 1
implies a negative relationship. This suggests that high-stress levels are significantly related to an
increased probability of having a second heart attack.
Thus, the odds ratio of 1.093 means that for each additional point on the stress scale, the odds of
having a second heart attack are increased 1.1 times. Or presented in percentage, for every
additional point on the stress scale, the odds of having a second heart attack increase by 9.3%.
The odds ratio of 0.130 regarding the ex ercise prescription variable can be interpreted as having a
second heart attack undergoing an exercise intervention is 0.13 times the odds of not having such an
intervention. Or presented in percentage, those having exercise prescription are 87% less likely to
have a second heart attack compared to those not having such an intervention.
The confusion matrix shows (where agreement (yes/yes) or disagreement (yes/no) occurred between
the predicted and observed) that the 15 true negative and positive cases were predicted by the model,
while the error, false negatives and positives, were found in 5 cases.  This is confirmed in the
Performance metrics, where both sensitivity (% of cases that had the outcome correctly predicted)
and specificity (% of cases correctly predicted as not having the outcome (i.e., true negatives) are both
75%.
PLOTS
These findings can be easily visualised through the inferential plots.

As stress levels increase, the probability of having a second heart attack increases.
No exercise intervention increases the probability of a 2nd heart attack, while it is reduced when it has
been put in place.

# UNDERSTANDING THE OUTPUT (page 94)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_94]

# REPORTING THE RESULTS (page 96)


Logistic regression was performed to ascertain the effects of stress and exercise intervention on the
likelihood that participants have a 2 nd heart attack. The logistic regression model was statistically
significant, 2 (37) = 21.257, p< .001. The model correctly classified 75.0% of cases. Increasing stress
was associated with an increased likelihood of a 2nd heart attack, but decreasing stress was associated
with a reduction in the likelihood. The presence of an exercise intervention programme reduced the
probability of a 2nd heart attack to 13%.

# REPORTING THE RESULTS (page 96)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_96]

# COMPARING MORE THAN TWO INDEPENDENT GROUPS (page 97)


ANOVA
Whereas t-tests compare the means of two groups/conditions, one-way analysis of variance (ANOVA)
compares the means of 3 or more groups/conditions. There are both independent and repeated
measures ANOVAs available in JASP. ANOVA has been described as an omnibus test which results in
an F-statistic that compares whether the dataset 's overall explained variance is significantly greater
than the unexplained variance. The null hypothesis tested is that  there is no significant difference
between the means of all the groups. If the null hypothesis is rejected, ANOVA just states that there
is a significant difference between the groups, but not where those differences occur.  To determine
where the group differences are, post hoc (From the Latin post hoc, "after this") tests are subsequently
used.
Why not just multiple pairwise comparisons? If there are 4 groups (A, B, C, D) , for example and the
differences were compared using multiple t-tests:
A vs. B  P<0.05   95% no type I error
A vs. C  P<0.05   95% no type I error
A vs. D  P<0.05   95% no type I error
B vs. C  P<0.05   95% no type I error
B vs. D  P<0.05   95% no type I error
C vs. D  P<0.05   95% no type I error
Assuming that each test was independent, the overall probability would be:
0.95 * 0.95 * 0.95 * 0.95 * 0.95 * 0.95 = 0.735
This is known as familywise error or, cumulative Type I error, and in this case , results in only a 73.5%
probability of no Type I error whereby the null hypothesis could be rejected when it is true. This is
overcome by using post hoc tests that make multiple pairwise comparisons with stricter acceptance
criteria to prevent familywise error.
ASSUMPTIONS
The independent ANOVA makes the same assumptions as most other parametric tests.
The independent variable must be categorical, and the dependent variable must be
continuous.
The groups should be independent of each other.
The dependent variable should be approximately normally distributed.
There should be no significant outliers.
There should be homogeneity of variance between the groups; otherwise, the p-value for the
F-statistic may not be reliable.
The first 2 assumptions are usually controlled using an appropriate research method design.
If the last three assumptions are violated, then the non-parametric equivalent, Kruskal-Wallis, should
be considered instead.

CONTRASTS
Contrasts are a priori tests (i.e. planned comparisons before any data were collected). As an
example, researchers may want to compare the effects of some new drugs to the currently
prescribed ones. These should only be a small set of comparisons to reduce family-wise error.
The choice must be based on the scientific questions being asked and chosen during the
experimental design. Hence, the term planned comparisons.   Therefore, they are looking at
specified mean differences and can be used if the ANOVA F test is insignificant.
JASP provides 6 planned contrasts enabling different types of comparisons:
Deviation: the mean of each level of the independent variable is compared to the overall
mean (the mean when all the levels are taken together).
Simple: the mean of each level is compared to the mean of a specified level, for example, with
the mean of the control group.
Difference: the mean of each level is compared to the mean of the previous levels.
Helmert: the mean of each level is compared to the mean of the subsequent levels.
Repeated: By selecting this contrast, the mean of each level is compared to the mean of the
following level.2
Polynomial: tests polynomial trends in the data.

# COMPARING MORE THAN TWO INDEPENDENT GROUPS (page 97)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_COMPARING_MORE_THAN_TWO_INDEPENDENT_GROUPS_PAGE_97]

# POST HOC TESTING (page 98)


Post hoc tests are tests that were decided upon after the data ha d been collected. They can only be
carried out if the ANOVA F test is significant.
JASP provides 4 types of post hoc testing
Standard
Games-Howell  used when you are unsure about the equality of group variances.
Dunnetts  used to compare all the groups to one group, i.e. the control group
Dunn  a non-parametric post hoc test used for testing small sub-sets of pairs.
JASP also provides 5 types of corrections for use with the independent group ANOVA tests:
Bonferroni  can be very conservative but gives guaranteed control over Type I error s at the risk of
reducing statistical power. Does not assume independence of the comparisons.
Holm  the Holm-Bonferroni test is a sequential Bonferroni method that is less conservative than the
original Bonferroni test.

Tukey  is one of the most commonly used tests and provides controlled Type I error for groups with
the same sample size and equal group variance.
Scheffe  controls for the overall confidence level when the group sample sizes are different.
Sidak  is similar to Bonferroni but assumes that each comparison is independent of the others .
Slightly more powerful than Bonferroni.

# POST HOC TESTING (page 98)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_98]

# EFFECT SIZE (page 99)


JASP provides 3 alternative effect size calculations for use with the independent group ANOVA tests:
Eta squared ( 2) - accurate for the sample variance explained but overestimates the population
variance. This can make it difficult to compare the effect of a single variable in different studies.
Partial Eta squared ( p2)  this solves the problem relating to population variance overestimation ,
allowing for comparison of the effect of the same variable in different studies.
Omega squared (2)  Normally, statistical bias gets very small as sample size increases, but for small
samples (n<30), 2 provides an unbiased effect size measure.
Test Measure Trivial Small Medium Large
ANOVA Eta
Partial Eta
Omega squared
<0.1
<0.01
<0.01
0.1
0.01
0.01
0.25
0.06
0.06
0.37
0.14
0.14

# EFFECT SIZE (page 99)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_EFFECT_SIZE_PAGE_99]

# RUNNING THE INDEPENDENT ANOVA (page 99)


Load Independent ANOVA diets.csv. This contains A column containing the 3 diets used (A, B and C)
and another column containing the absolute amount of weight loss after 8 weeks on one of 3 different
diets.  For good practice, check the descriptive statistics and the boxplots for any extreme outliers.
Go to ANOVA > ANOVA, put weight loss into the Dependent Variable and the Diet groupings into the
Fixed Factors box.  In the first instance, tick Descriptive statistics and 2 as the effect size.

In Assumptions checks, tick all options:
This should result in 3 tables and one Q-Q plot.

# RUNNING THE INDEPENDENT ANOVA (page 99)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_INDEPENDENT_ANOVA_PAGE_99]

# UNDERSTANDING THE OUTPUT (page 100)


The main ANOVA table shows that the F-statistic is significant (p<.001) and that there is a large effect
size. Therefore, there is a significant difference between the means of the 3 diet groups.

# UNDERSTANDING THE OUTPUT (page 100)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_100]

# TESTING ASSUMPTIONS (page 100)


Before accepting this, any violations in the assumptions required for an ANOVA should be checked.
Levenes test shows that the homogeneity of variance is not significant . However, if Levenes test
shows a significant difference in variance, the Brown -Forsythe or Welch correction should be
reported.

The Q-Q plot shows that the data appear to be normally distributed and linear.
The descriptive statistics suggest that Diet 3 results in the highest weight loss after 8 weeks.

# TESTING ASSUMPTIONS (page 100)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TESTING_ASSUMPTIONS_PAGE_100]

# CONTRAST EXAMPLE (page 101)


If, for example, one planned to compare the effects of diets B and C to diet A, click on the drop-
down menu and select simple next to diet. This will test the significance between the first category
in the list with the remaining categories.
As can be seen, only diet C is significantly different from diet A (t(69) = 4.326, p<.001.

If the ANOVA reports no significant difference, you can go no further in the analysis.

# CONTRAST EXAMPLE (page 101)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_CONTRAST_EXAMPLE_PAGE_101]

# POST HOC TESTING (page 102)


If the ANOVA is significant, post hoc testing can now be carried out. In Post Hoc Tests, add Diet to the
analysis box on the right, tick Standard type, use Tukey for the post hoc correction  and tick the flag
significant comparisons.
Post hoc testing shows that there is no significant difference between weight loss in diets A and B.
However, It is significantly higher in diet C compared to diet A (p<.001) and diet B (p=.001). Cohens d
shows that these differences have a large effect size.
Also, in Descriptive Plots, add the Factor Diet to the horizontal axis and tick display error bars and in
Raincloud plots, do the same but tick Horizontal display.

# POST HOC TESTING (page 102)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_102]

# REPORTING THE RESULTS (page 103)


Levenes test showed that there was equal variance for all three groups: F (2,69) = 1.3, p=.28*.
Independent one-way ANOVA showed a significant effect of the type of diet on weight loss after 10
weeks (F (2, 69) =46.184, p<.001, The 2 = 0.214, suggesting that this is a large effect size.
Post hoc testing using pairwise comparisons of the estimated marginal means with Tukeys correction
revealed that diet C resulted in significantly greater weight loss (M = 5.59, SD = 2.1 kg) than diet A (M
= 3.0, SD = 2.36 kg, p<.001) or diet B ( M = 3.41, SD = 2.36 kg, p=.001).  There were no significant
differences in weight loss between diets A and B (p=.777).
* If Levenes test shows a significant difference in variance, the Brown -Forsythe or Welch
correction should be reported.

# REPORTING THE RESULTS (page 103)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_103]

# KRUSKAL-WALLIS – NON-PARAMETRIC ANOVA (page 104)


If your data fails parametric assumption tests or is nominal, the Kruskal -Wallis H test is a non -
parametric equivalent to the independent samples ANOVA. It can be used for comparing two or more
independent samples of equal or different sample sizes.  Like the Mann-Whitney and Wilcoxon tests,
it is a rank-based test.
As with the ANOVA, the Kruskal-Wallis H test ( also known as the "one-way ANOVA on ranks" ) is an
omnibus test that does not specify which specific groups of the independent variable are statistically
significantly different from each other . To do this, JASP provides the option for running Dunns post
hoc test. These multiple comparison tests can be very conservative, particularly for large numbers of
comparisons.
Load the Kruskal-Wallis ANOVA.csv dataset into JASP. This dataset contains subjective pain scores for
participants undergoing no treatment (control), cryotherapy or combined cryotherapy -compression
for delayed onset muscle soreness after exercise.

# KRUSKAL-WALLIS – NON-PARAMETRIC ANOVA (page 104)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_KRUSKAL_WALLIS_NON_PARAMETRIC_ANOVA_PAGE_104]

# RUNNING THE KRUSKAL-WALLIS TEST (page 104)


Go to ANOVA >ANOVA. In the analysis window, add the Pain score to the dependent variable and
treatment to the fixed factors.  Check that the pain score is set to ordinal. This will automatically run
the normal independent ANOVA.  Under Assumption Checks, tick both Homogeneity tests and Q -Q
plots.

Although the ANOVA indicates a significant result, the data ha ve not met the assumptions of
homogeneity of variance as seen by the significant Levenes test and only shows linearity in the middle
of the Q-Q plot and curves off at the extremities, indicating more extreme values.  Added to the fact
that the dependent variable is based on subjective pain scores suggest s the use of a non-parametric
alternative.
Return to the statistics options and open the Nonparametric option at the bottom. For the Kruskal -
Wallis test, move the Treatment variable to the box on the right and tick Dunns post hoc test.

# RUNNING THE KRUSKAL-WALLIS TEST (page 104)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_KRUSKAL_WALLIS_TEST_PAGE_104]

# UNDERSTANDING THE OUTPUT (page 105)


Two tables are shown in the output. The Kruskal-Wallis test shows that there is a significant difference
between the three treatment modalities.
Dunns post hoc test provides its p-value as well as those for Bonferroni and Holms Bonferroni
correction. As can be seen, both treatment conditions are significantly different from the controls, but
not from each other.

# UNDERSTANDING THE OUTPUT (page 105)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_105]

# REPORTING THE RESULTS (page 106)


Pain scores were significantly affected by treatment modality , H(2) = 19.693, p<.001. Pairwise
comparisons using Dunns post hoc showed that both cryotherapy (Mdn = 3) and cryotherapy with
compression (Mdn = 3) significantly reduced pain scores (p=.002 and p<.001, respectively) compared
to the control group  (Mdn = 7) . There w as no significant difference between cryotherapy and
cryotherapy with compression (p=.102).

# REPORTING THE RESULTS (page 106)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_106]

# COMPARING MORE THAN TWO RELATED GROUPS (page 107)


RMANOVA
The one-way repeated measures ANOVA ( RMANOVA) is used to assess if there is a difference in
means between 3 or more groups (where the participants are the same in each group) that have been
tested multiple times or under different conditions. Such a research design, for example, could be that
the same participants were tested for an outcome measure at 1, 2 and 3 weeks or that the outcome
was tested under conditions 1, 2 and 3.
The null hypothesis tested is that  there is no significant difference between the means of the
differences between all the groups.
The independent variable should be categorical, and the dependent variable needs to be a continuous
measure. In this analysis , the independent categories are termed levels, i.e. these are the related
groups. So in the case where an outcome was measured at weeks 1, 2 and 3, the 3 levels would be
week 1, week 2 and week 3.
The F-statistic is calculated by dividing the mean squares for the variable (variance explained by the
model) by its error mean squares (unexplained variance). The larger the F -statistic, the more likely it
is that the independent variable will have had a significant effect on the dependent variable.
ASSUMPTIONS
The RMANOVA makes the same assumptions as most other parametric tests.
The dependent variable should be approximately normally distributed.
There should be no significant outliers.
Sphericity relates to the equality of the variances of the differences between levels of the
repeated measures factor.
If the assumptions are violated , then the non -parametric equivalent, Friedmans test, should be
considered instead and is described later in this section.
SPHERICITY
If a study has 3 levels (A, B and C), sphericity assumes the following:
Variance (A-B)  Variance (A-C)  Variance (B-C)
RMANOVA checks the assumption of sphericity using Mauchlys (pronounced Mockleys) test of
sphericity. This tests the null hypothesis that the variances of the differences are equal .  In many
cases, repeated measures violate the assumption of sphericity , which can lead to Type I error. If this
is the case, corrections to the F-statistic can be applied.
JASP offers two methods of correcting the F -statistic, the Greenhouse-Geisser and the Huynh-Feldt
epsilon () corrections. A general rule of thumb is that if the  values are <0.75 , then use the
Greenhouse-Geisser correction and if they are >0.75 then use the Huynh-Feldt correction.

# COMPARING MORE THAN TWO RELATED GROUPS (page 107)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_COMPARING_MORE_THAN_TWO_RELATED_GROUPS_PAGE_107]

# POST HOC TESTING (page 108)


Post hoc testing is limited in RMANOVA. JASP provides two alternatives:
Bonferroni  can be very conservative but gives guaranteed control over Type I error s at the risk of
reducing statistical power.
Holm  the Holm-Bonferroni test is a sequential Bonferroni method that is less conservative than the
original Bonferroni test.
If you ask for either Tukey or Scheffe post hoc corrections, JASP will return a NaN (not a number) error.

# POST HOC TESTING (page 108)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_108]

# EFFECT SIZE (page 108)


JASP provides the same alternative effect size calculations that are used with the independent group
ANOVA tests:
Eta squared ( 2) - accurate for the sample variance explained , but overestimates the population
variance. This can make it difficult to compare the effect of a single variable in different studies.
Partial Eta squared ( p2)  this solves the problem relating to population variance overestimation ,
allowing for comparison of the effect of the same variable in different studies. This appears to be the
most commonly reported effect size in repeated measures ANOVA.
Omega squared (2)  Normally, statistical bias gets very small as sample size increases, but for small
samples (n<30), 2 provides an unbiased effect size measure.
Levels of effect size:
Test Measure Trivial Small Medium Large
ANOVA Eta
Partial Eta
Omega squared
<0.1
<0.01
<0.01
0.1
0.01
0.01
0.25
0.06
0.06
0.37
0.14
0.14

# EFFECT SIZE (page 108)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_EFFECT_SIZE_PAGE_108]

# RUNNING THE REPEATED MEASURES ANOVA (page 108)


Load Repeated ANOVA cholesterol.csv . This contains one column with the participant IDs and 3
columns, one for each repeated measurement of blood cholesterol following an intervention.  For
good practice, check the descriptive statistics and the boxplots for any extreme outliers.
Go to ANOVA > Repeated measures ANOVA. As stated above, the independent variable (repeated
measures factor) has levels ; in this case, there are 3 levels.  Rename RM Factor 1 to Time post -
intervention, and then rename 3 levels to Week 0, Week 3 and Week 6 accordingly.

Once these have been done , they will appear in the Repeated Measures Cells. Now add the
appropriate data to the appropriate level.
Tick Descriptive Statistics, Estimates of effect size and 2.
Under Assumption Checks, tick Sphericity tests and all Sphericity correction options.
The output should consist of 4 tables. The third table, between -subject effects, can be ignored for
this analysis.

# RUNNING THE REPEATED MEASURES ANOVA (page 108)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_REPEATED_MEASURES_ANOVA_PAGE_108]

# UNDERSTANDING THE OUTPUT (page 110)


The within-subjects effects table reports a large F-statistic, which is highly significant (p<.001) and has
a small to medium effect size (0.058). This table shows the statistics for sphericity assumed (none) and
the two correction methods. The main differences are in the degrees of freedom (df) and the value of
the mean square.  Under the table, it is noted that the assumption of sphericity has been violated.
The following table gives the results of Mauchlys test of sphericity. There is a significant difference
(p<.001) in the variances of the differences between the groups. Greenhouse-Geisser and the Huynh-
Feldt epsilon () values are below 0.75. Therefore, the ANOVA result should be reported based on the
Greenhouse-Geisser correction:
To provide a cleaner table, go back to Assumption Checks and only tick Greenhouse -Geisser for
sphericity correction.
There is a significant difference between the means of the differences between all the groups , F
(1.235, 21.0) =212.3, p<.001, 2 = 0.058.

The descriptive data suggest that blood cholesterol levels were higher at week 0 compared to weeks
3 and 6.
However, if the ANOVA reports no significant difference, you can go no further in the
analysis.

# UNDERSTANDING THE OUTPUT (page 110)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_110]

# POST HOC TESTING (page 111)


If the ANOVA is significant, post hoc testing can now be carried out. In Post Hoc Tests, add Time post-
intervention to the analysis box on the right, tick Effect size , and, in this case, use Holm for the post
hoc correction.

Also, in Descriptive Plots, add the Factor  Time post-intervention to the horizontal axis and tick display
error bars.
Post hoc testing shows that there are significant differences in blood cholesterol levels between all of
the time point combinations and are associated with large effect sizes.

# POST HOC TESTING (page 111)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_111]

# REPORTING THE RESULTS (page 112)


Mauchlys test of sphericity showed a significant difference in variation between the group
differences: W (2) = 0.381, p<.001. Therefore, the Greenhouse -Geisser correction was used. This
showed that cholesterol levels differed significantly between  time points, F(1.235, 21.0) =212.3,
p<.001.  2 = 0.058, suggesting that this is a small to medium effect.
Post hoc testing using the Holm correction revealed that cholesterol levels decreased significantly as
time increased, weeks 0  3 (M = 0.566, SE = 0.04, p<.001), weeks 1  6 (M = 0.63 SE = 0.04, p=.004)
and weeks 3-6 (M=0.06, SE = 0.016).

# REPORTING THE RESULTS (page 112)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_112]

# FRIEDMAN’S REPEATED MEASURES ANOVA (page 113)


If parametric assumptions are violated or the data is ordinal , you should consider using the non -
parametric alternative, Friedmans test. Similar to the Kruskal-Wallis test, Friedmans test is used for
one-way repeated measures analysis of variance by ranks and doesnt assume the data comes from a
particular distribution. This test is another omnibus test that does not specify which specific groups of
the independent variable are statistically significantly different from each other . To do this, JASP
provides the option for running Conovers post hoc test if Friedmans test is significant.
Load Friedman RMANOVA.csv into JASP.  This has 3 columns of subjective pain ratings measured at
18-, 36- and 48-hours post-exercise. Check that the pain scores are set to ordinal data.

# FRIEDMAN’S REPEATED MEASURES ANOVA (page 113)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_FRIEDMAN_S_REPEATED_MEASURES_ANOVA_PAGE_113]

# RUNNING THE FRIEDMAN’S TEST (page 113)


Go to ANOVA > Repeated measures ANOVA. The independent variable (repeated measures factor)
has 3 levels.  Rename RM Factor 1 to Time, and then rename the 3 levels to 18 hours, 36 hours and 48
hours accordingly.
Once these have been done, they will appear in the Repeated Measures Cells. Now add the
appropriate dataset to the appropriate level.
This will automatically produce the standard repeated measures within-subjects ANOVA table.  To run
a Friedmans test, expand the Nonparametrics tab, move Time to the RM factor box and tick Conovers
post hoc tests.

# RUNNING THE FRIEDMAN’S TEST (page 113)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_FRIEDMAN_S_TEST_PAGE_113]

# UNDERSTANDING THE OUTPUT (page 114)


Two tables should be produced.
Friedmans test shows that time has a significant effect on pain perception. Connors post hoc pairwise
comparisons show that all pain perception is significantly different between each time point.

# UNDERSTANDING THE OUTPUT (page 114)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_114]

# REPORTING THE RESULTS (page 115)


Friedmanns test showed that time has  a significant effect on subjective pain scores 2F (2) = 26.77,
p<.001. Conovers pairwise post hoc comparisons showed that pain perception peaked at 36 hours
(Mdn = 7) and was significantly higher tha n at 18 hours ( Mdn = 3: T(28) = 15.17, p<.001) and then
decreased by 48 hours (Mdn = 3: T(28) = 8.82 p<.001).

# REPORTING THE RESULTS (page 115)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_115]

# COMPARING INDEPENDENT GROUPS AND THE EFFECTS OF COVARIATES (page 116)


ANCOVA
ANOVA can be used to compare the means of one variable (dependent) in two or more groups,
whereas analysis of covariance (ANCOVA) sits between ANOVA and regression and compares the
means of one (dependent) variable in two or more groups  while considering the variability of other
continuous variables (COVARIATES). ANCOVA checks for differences in adjusted means (i.e. adjusted
for the effects of the covariate). A covariate may not usually be part of the main research question ,
but could influence the dependent variable and therefore needs to be adjusted or controlled for.  If a
good covariate is used, ANCOVA will have improved statistical power and control over error.
Control for   to subtract statistically the effects of a variable (a control variable) to see what a
relationship would be without it (Vogt 1977).
Hold constant  to subtract the effects of a variable from a complex relationship to study what the
relationship would be if the variable were, in fact, a constant. Holding a variable constant essentially
means assigning it an average value (Vogt 1977).
Statistical control  using statistical techniques to isolate or subtract variance in the dependent
variable attributable to variables that are not the subject of the study (Vogt, 1999).
For example, when looking for a difference in weight loss between the three diets , it would be
appropriate to take into account the individuals ' pre-trial bodyweight since heavier people may lose
proportionately more weight.
Type of diet
(Factor)
Weight loss
Starting body weight
(Covariate)
Independent
variables
Dependent
variable
ANOVA
ANCOVA

The null hypothesis tested is that there is no significant difference between the adjusted means of
all the groups.
ASSUMPTIONS
ANCOVA makes the same assumptions as the independent ANOVA . However, there are two further
assumptions:
The relationship between the dependent and covariate variables is linear.
Homogeneity of regression, i.e. the regression lines for each of the independent groups are
parallel to each other.

# COMPARING INDEPENDENT GROUPS AND THE EFFECTS OF COVARIATES (page 116)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_COMPARING_INDEPENDENT_GROUPS_AND_THE_EFFECTS_OF_COVARIATES_PAGE_116]

# POST HOC TESTING (page 117)


JASP provides 4 Types
Standard  as above
Games-Howell  used when you are unsure about the equality of group variances.
Dunnetts  used to compare all the groups to one group, i.e. the control group
Dunn  a non-parametric post hoc test used for testing small sub-sets of pairs.
JASP additionally provides 4 corrections for use with the independent group ANOVA tests:
Bonferroni  can be very conservative but gives guaranteed control over Type I error s at the
risk of reducing statistical power.
Holm  the Holm-Bonferroni test is a sequential Bonferroni method that is less conservative
than the original Bonferroni test.
Tukey  is one of the most commonly used tests and provides controlled Type I error for
groups with the same sample size and equal group variance.
Scheffe  controls for the overall confidence level when the group sample sizes are different.
Covariate
Dependent variable
Diet 1
Diet 2
Diet 3
Covariate
Dependent variable
Diet 1
Diet 2
Diet 3
Homogeneity of regression
Assumption violated
Homogeneity of regression

# POST HOC TESTING (page 117)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_117]

# EFFECT SIZE (page 118)


JASP provides 3 alternative effect size calculations for use with the independent group ANOVA tests:
Eta squared ( 2) - accurate for the sample variance explained , but overestimates the
population variance. This can make it difficult to compare the effect of a single variable in
different studies.
Partial Eta squared ( p2)  this solves the problem relating to population variance
overestimation, allowing for comparison of the effect of the same variable in different studies.
Omega squared (2)  Normally, statistical bias gets very small as sample size increases, but
for small samples (n<30), 2 provides an unbiased effect size measure.
Test Measure Trivial Small Medium Large
ANOVA Eta
Partial Eta
Omega squared
<0.1
<0.01
<0.01
0.1
0.01
0.01
0.25
0.06
0.06
0.37
0.14
0.14

# EFFECT SIZE (page 118)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_EFFECT_SIZE_PAGE_118]

# RUNNING THE INDEPENDENT ANCOVA (page 118)


Load ANCOVA hangover.csv. This dataset has been adapted from the one provided by Andy Field
(2017).  The morning after a Freshers ball , students were given either water, coffee or a Barocca to
drink.  Two hours later , they reported how well they felt (from 0  awful to 10 very well).  At the
same time, data were collected on how drunk they were the night before (0-10).
Initially, run an ANOVA with wellness as the dependent variable and the type of drink as the fixed
factor.
As can be seen from the results, homogeneity of variance has not been violated , while the ANOVA
shows that there is no significant difference in the wellness scores between any of the morning drinks.
F(2,27)=1.714, p=.199. However, this may be related to how drunk the students were the night before!
Go to ANOVA > AN COVA, put wellness as the dependent variable and the type of drink as the fixed
factor. Now add drunkenness to the Covariate(s) box.   In the first instance, tick Descriptive statistics
and 2 as the effect size;

In Assumption Checks, tick both options.
In Marginal Means, move drink to the right.
This should result in 4 tables and one Q-Q plot.

# RUNNING THE INDEPENDENT ANCOVA (page 118)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_INDEPENDENT_ANCOVA_PAGE_118]

# UNDERSTANDING THE OUTPUT (page 120)


The covariate (drunkenness) significantly predicts wellness (p<.001). The effects of the type of drink
on wellness, when adjusted for the effects of drunkenness, are now significant (p=.003).
Levenes test is significant, unlike in ANOVA, where no homogeneity of variance corrections (i.e.
Welch) are provided. For ANCOVA, this can be ignored. The Q-Q plot appears to be normal.

The descriptive statistics show the unadjusted means for wellness in the three drink groups.
The marginal means are now the wellness means, having been adjusted for the effects of the covariate
(drunkenness).

# UNDERSTANDING THE OUTPUT (page 120)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_120]

# TESTING FURTHER ASSUMPTIONS (page 121)


As previously mentioned, the assumption of homogeneity of regression is important in ANCOVA. This
can be tested by looking at the interaction between the type of drink and the drunkenness scores. Go
to Model, drink, and drunkenness will have been automatically added as individual Model terms. Now
highlight both drink and drunkenness and add them both to Model terms.
The ANOVA table now has an extra row showing the interaction between the type of drink and
drunkenness. This is not significant (p=.885), i.e. the relationships between drunkenness and wellness
are the same in each drink group.  If this is significant , there will be concerns over the validity of the
main ANCOVA analysis.
Having checked this, go back and remove the interaction term from the Model terms.
If the ANCOVA reports no significant difference, you can go no further in the analysis.

# TESTING FURTHER ASSUMPTIONS (page 121)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TESTING_FURTHER_ASSUMPTIONS_PAGE_121]

# POST HOC TESTING (page 122)


If the ANCOVA is significant, post hoc testing can now be carried out. In Post Hoc Tests , add Drink to
the analysis box on the right, tick Effect size , and, in this case, use Tukey for the post hoc correction.
Also, tick flag significant comparisons.
Post hoc testing shows that there is no significant difference between coffee and water on wellness.
However, wellness scores were significantly higher after drinking a Barocca.
This can be seen from the Descriptive plots.

# POST HOC TESTING (page 122)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_122]

# REPORTING THE RESULTS (page 123)


The covariate, drunkenness, was significantly related to the morning after wellness, F(1,26) = 33.03,
p<.001, 2 = 0.427.  There was also a significant effect of the drink on wellness after controlling for
drunkenness, F(2, 26) = 7.47, p=.003, 2 = 0.173.
Post hoc testing using Tukeys correction revealed that drinking a Barocca  resulted in significantly
greater wellness compared to water  (p=.004) and compared to coffee ( p=.01. There were no
significant differences in wellness between water and coffee (p=.973).

# REPORTING THE RESULTS (page 123)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_123]

# TWO-WAY INDEPENDENT ANOVA (page 124)


One-way ANOVA tests situations when only one independent variable is manipulated, and two-way
ANOVA is used when more than 1 independent variable has been manipulated.  In this case,
independent variables are known as factors.

# TWO-WAY INDEPENDENT ANOVA (page 124)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TWO_WAY_INDEPENDENT_ANOVA_PAGE_124]

# FACTOR 1 FACTOR 2 (page 124)


CONDITION 1 Group 1 Dependent variable
Group 2 Dependent variable
CONDITION 2 Group 1 Dependent variable
Group 2 Dependent variable
CONDITION 3 Group 1 Dependent variable
Group 2 Dependent variable
The factors are split into levels; therefore, in this case, Factor 1 has 3 levels and Factor 2 has 2 levels.
A main effect is the effect of one of the independent variables on the dependent variable, ignoring
the effects of any other independent variables. There are 2 main effects tested , both of which are
between-subjects: in this case , comparing differences between factor 1 (i.e. condition) and
differences between factor 2 (i.e. groups). Interaction is where one factor influences the other factor.
The two-way independent ANOVA is another omnibus test that is used to test 2 null hypotheses:
1. There is no significant between-subject effect, i.e. no significant difference between the
means of the groups in either of the factors.
2. There is no significant interaction effect , i.e. no significant group differences across
conditions.
ASSUMPTIONS
Like all other parametric tests, mixed factor ANOVA makes a series of assumptions that should either
be addressed in the research design or can the tested for.
The independent variables (factors) should have at least two categorical independent groups
(levels).
The dependent variable should be continuous and approximately normally distributed for all
combinations of factors.
There should be homogeneity of variance for each of the combinations of factors.
There should be no significant outliers.

# FACTOR 1 FACTOR 2 (page 124)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_FACTOR_1_FACTOR_2_PAGE_124]

# RUNNING TWO-WAY INDEPENDENT ANOVA (page 124)


Open 2-way independent ANOVA.csv in JASP. This comprises 3 columns of data, Factor 1  gender
with 2 levels (male and female), Factor 2 - supplement with 3 levels (control, carbohydrate CHO and
protein) and the dependent variable (explosive jump power. In Descriptive statistics, check the data
for significant outliers.  Go to ANOVA >ANOVA, add Jump power to the Dependent variable, Gender
and Supplement to the Fixed factors.

Tick Descriptive statistics and Estimates of effect size (2).
In Descriptive plots , add the supplement to the horizontal axis and Gender to separate lines. In
Additional Options,

# RUNNING TWO-WAY INDEPENDENT ANOVA (page 124)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_TWO_WAY_INDEPENDENT_ANOVA_PAGE_124]

# UNDERSTANDING THE OUTPUT (page 126)


The output should comprise 2 tables and one plot.
The ANOVA table shows that there are significant main effects for both Gender and Supplement
(p=0.003 and p<.001, respectively) with medium and large effect sizes, respectively.  This suggests that
there is a significant difference in jump power between genders, irrespective of Supplement, and
significant differences between supplements, irrespective of Gender.
There is also a significant interaction between Gender and Supplement ( p<.001), which also has a
medium to large effect size (0.138). This suggests that the differences in jump power between genders
are affected somehow by the type of supplement used.
The Descriptive statistics and plot suggest that the main differences are between genders when using
a protein supplement.

# UNDERSTANDING THE OUTPUT (page 126)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_126]

# TESTING ASSUMPTIONS (page 127)


In Assumption Checks, tick Homogeneity tests and Q-Q plot of residuals.
Levenes test shows no significant difference in variance within the dependent variable groups; thus,
homogeneity of variance has not been violated.
The Q-Q plot shows that the data appear to be normally distributed and linear. We can now accept
the ANOVA result since none of these assumptions has been violated.
However, if the ANOVA reports no significant difference, you can go no further with the
analysis.

# TESTING ASSUMPTIONS (page 127)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TESTING_ASSUMPTIONS_PAGE_127]

# SIMPLE MAIN EFFECTS (page 127)


Go to the analysis options and Simple Main Effects. Here , add Gender to the Simple effect
factor and Supplement to the Moderator Factor 1. Simple main effects are effectively limited
to pairwise comparisons.

This table shows that there are no gender differences in jump power between the control and
CHO groups ( p=.116 and p=0.058, respectively). However, there is a significant difference
(p<.001) in jump power between genders in the protein supplement group.

# SIMPLE MAIN EFFECTS (page 127)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_SIMPLE_MAIN_EFFECTS_PAGE_127]

# CONTRAST TESTING (page 128)


There are two ways of testing for a difference between (combinations of) cells: post hoc tests
and contrast analysis. JASP has a range of different contrast tests available, including custom
contrasts. For example, we can contrast the three different supplements.  Open up the
Contrasts menu, and next to Supplement , click on the drop -down menu and select custom.
This will add another series of options to this window.

In this window, contrasts can be added; in this case, three contrasts can be defined:
Contrast 1 = CHO vs Protein
Contrast 2 = CHO vs Control
Contrast 3 = Control vs Protein

This will result in the following tables:
Comparing this table to the post hoc analysis below , the estimates of the differences in
marginal means are the same, as well as their standard errors and t-statistics. However, both
the p-values and confidence intervals vary: the corrected p -values are typically higher, and
the confidence intervals are wider, for the post hoc analysis.

# CONTRAST TESTING (page 128)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_CONTRAST_TESTING_PAGE_128]

# POST HOC TESTING (page 130)


If the ANOVA is significant, post hoc testing can now be carried out. In Post Hoc Tests, add Supplement
and the Gender*Supplement  to the analysis box on the right, tick Effect size , and, in this case, use
Tukey for the post hoc correction. Also, for ease of viewing, tick Flag significant comparisons.
Post hoc testing is not done for Gender since there are only 2 levels.
Post hoc testing shows no significant difference between the control and CHO, supplement group,
irrespective of Gender, but significant differences between Control and Protein (p<.001) and between
CHO and Protein (p<.001).

The post hoc comparisons for the interactions decompose the results further:

# POST HOC TESTING (page 130)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_130]

# REPORTING THE RESULTS (page 131)


A two-way ANOVA was used to examine the effect of gender and supplement type on explosive jump
power. There were significant main effects for both gender ( F (1, 42) = 9.59, p=.003, 2 = 0.058) and
Supplement (F (2, 42) = 3 6.1, p<.001, 2 = 0.477).  There was a statistically significant interaction
between the effects of gender and supplement on explosive jump power (F (2, 42) = 11.1, p<.001, 2
= 0.138).
Tukeys post hoc correction showed that explosive leg power was significantly higher in the protein
group compared to the control or CHO groups (t=--7.52, p<.001 and t=--7.06, p<.001 respectively).
Simple main effects showed that jump power was significantly higher in males on a protein
supplement (M = 1263, SD = 140) compared to females (M = 987, SD = 92, F (1) =26.06, p<.001).

# REPORTING THE RESULTS (page 131)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_131]

# TWO-WAY REPEATED MEASURES ANOVA (page 132)


Two-way repeated Measures ANOVA means that there are two factors in the experiment, for example,
different treatments and different conditions. "Repeated  measures" means that the same subject
received more than one treatment and/or more than one condition.
Independent
variable (Factor 2)
Independent variable (Factor 1) = time
Participant Time 1 Time 2 Time 3
Condition 1 1 Dependent
variable
Dependent
variable
Dependent
variable
2 Dependent
variable
Dependent
variable
Dependent
variable
3 Dependent
variable
Dependent
variable
Dependent
variable
Condition 2 1 Dependent
variable
Dependent
variable
Dependent
variable
2 Dependent
variable
Dependent
variable
Dependent
variable
3 Dependent
variable
Dependent
variable
Dependent
variable
The factors are split into levels; therefore, in this case, Factor 1 has 3 repeated levels and Factor 2 has
2 repeated levels.
A main effect is the effect of one of the independent variables on the dependent variable, ignoring
the effects of any other independent variables. There are 2 main effects tested , both of which are
between-subjects: in this case , comparing differences between factor 1 (i.e. condition) and
differences between factor 2 (i.e. groups). Interaction is where one factor influences the other factor.
The two-way repeated ANOVA is another omnibus test that is used to test the following main effect
null hypotheses:
H01: The dependent variable scores are the same for each level in factor 1 (ignoring factor 2).
H02: The dependent variable scores are the same for each level in factor 2 (ignoring factor 1).
The null hypothesis for the interaction between the two factors is:
H03: The two factors are independent, or that an interaction effect is not present.
ASSUMPTIONS
Like all other parametric tests, two-way repeated ANOVA makes a series of assumptions that should
either be addressed in the research design or can the tested for.
The independent variables (factors) should have  at least two categorical  related groups
(levels).

The dependent variable should be continuous and approximately normally distributed for all
combinations of factors.
Sphericity, i.e. the variances of the differences between all combinations of related groups
must be equal.
There should be no significant outliers.

# TWO-WAY REPEATED MEASURES ANOVA (page 132)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TWO_WAY_REPEATED_MEASURES_ANOVA_PAGE_132]

# RUNNING TWO-WAY REPEATED MEASURES ANOVA (page 133)


Open 2-way repeated ANOVA.csv in JASP. This comprises 4 columns of data (sit and reach flexibility
scores for two factors, Factor 1 with 2 levels (stretch and no stretch) and Factor 2 with 2 levels (warm-
up and no warm-up). In Descriptive statistics, check the data for significant outliers.  Go to ANOVA >
Repeated Measures ANOVA.  Firstly, each Factor and its levels should be defined . For RM Factor 1
define this as Stretching and its levels as stretch and no stretch. Then define RM Factor 2 as Warm-up
and its levels as warm-up and no warm-up.  Then add the appropriate column of data to the assigned
repeated measures cells.
Also, tick Descriptive statistics and estimates of effect size - 2.

In Descriptive plots, add the Stretching factor to the horizontal axis and the Warm-up factor
to separate lines. Tick the display error bars option.

# RUNNING TWO-WAY REPEATED MEASURES ANOVA (page 133)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_TWO_WAY_REPEATED_MEASURES_ANOVA_PAGE_133]

# UNDERSTANDING THE OUTPUT (page 134)


The output should comprise 3 tables and one plot. The Between-Subjects Effects table can be ignored
in this analysis.
The ANOVA within-subjects effects table shows that there are significant main effects for both stretch
(p<.001) and warm-up (p<.001) on sit and reach distance. Both are associated with large effect sizes.
There is also a significant interaction between stretch and warm-up (p<.001), which suggests that the
effects of performing a stretch on sit and reach distance are different depending on whether a warm-
up had been performed.  These findings can be seen in both the descriptive data and the plot.

# UNDERSTANDING THE OUTPUT (page 134)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_134]

# TESTING ASSUMPTIONS (page 135)


In this case, there are no assumption checks.  Sphericity can only be tested when there are at
least three levels, and homogeneity requires at least two unrelated data sets.  If a factor has
more than 2 levels , Mauchlys test of Sphericity should also be run, and the appropriate
corrected F value used if necessary (See Repeated Measures ANOVA for details).
However, if the ANOVA reports no significant difference, you can go no further with the
analysis.

# TESTING ASSUMPTIONS (page 135)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TESTING_ASSUMPTIONS_PAGE_135]

# SIMPLE MAIN EFFECTS (page 136)


Now go to the analysis options and Simple Main Effects. Here, add Warm up to the Simple effect factor
and Stretch to the Moderator Factor 1. Simple main effects are effectively pairwise comparisons.
This table shows that when moderating for warm -up there is a significant difference ( p<.001) in sit
and reach performance when a stretch was also carried out but not without a stretch (p=.072).

We can now moderate for stretch by changing the Simple Main Effects to use Stretch as the simple
effect factor and warm-up as the moderator factor. We can also replot the descriptives with a warm-
up on the horizontal axis and stretch as separate lines.
In this case, when controlling for Stretch , there were significant differences between both warm -up
and no warm-up.
Both simple main effects can be visualised in their descriptive plots.
The data can be further visualised using Raincloud plots.

# SIMPLE MAIN EFFECTS (page 136)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_SIMPLE_MAIN_EFFECTS_PAGE_136]

# POST HOC TESTING (page 138)


If the ANOVA is significant , post hoc testing can now be carried out. In Post Hoc Tests , add stretch,
warm-up and the Stretching*warm-up interaction to the analysis box on the right, tick Effect size, and,
in this case, use Holm for the post hoc correction. Tick Flag significant comparisons.

Post hoc testing for the main effects confirms that there are significant differences in sit and reach
distance when comparing the two levels of each factor.  This is further decomposed in the Post hoc
comparisons for the interaction.

# POST HOC TESTING (page 138)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_138]

# REPORTING THE RESULTS (page 140)


A two -way ANOVA was used to examine the effect of stretch and warm-up type on sit and teach
performance. There were significant main effects for stretch (F (1, 11) = 123.4, p<.001, 2 = 0.647) and
warm-up (F (1, 11) = 68.69, p<.001,  2 = 0.4 04).  There was a statistically significant interaction
between the effects of stretch and warm-up on sit and reach performance (F (1, 11) = 29.64, p<.001,
2 = 0.215).
Simple main effects showed that sit and reach performance  was significantly higher when both a
stretch and warm-up had been done (F (1) =234, p<.001).

# REPORTING THE RESULTS (page 140)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_140]

# MIXED FACTOR ANOVA (page 141)


Mixed factor ANOVA (another two -way ANOVA) is a combination of both independent and
repeated measures ANOVA involving more than 1 independent variable (known as factors).
The factors are split into levels; therefore, in this case, Factor 1 has 3 levels and Factor 2 has
2 levels. This results in 6 possible combinations.
A main effect is the effect of one of the independent variables on the dependent variable,
ignoring the effects of any other independent variables . There are 2 main effects tested: in
this case, comparing data across factor 1 (i.e. time) is known as the  within-subjects factor,
while comparing differences between factor 2 (i.e. groups) is known as the  between-
subjects factor. Interaction is where one factor influences the other factor.
The main effect of time or condition tests the following, i.e. irrespective of which group is in
Independent variable
(Factor 2)
Independent variable (Factor 1) = time or condition
Time/condition 1 Time/condition 2 Time/condition 3
Group 1 All data All data All data Group 2
The main effect of group tests is the following, i.e. irrespective of which condition the data is
in:
Independent variable
(Factor 2)
Independent variable (Factor 1) = time or condition
Time/condition 1 Time/condition 2 Time/condition 3
Group 1 All data
Group 2 All data
Simple main effects are effectively pairwise comparisons:
Independent variable
(Factor 2)
Independent variable (Factor 1) = time or condition
Time/condition 1 Time/condition 2 Time/condition 3
Group 1 Data Data Data
Group 2 Data Data Data
Independent variable
(Factor 2)
Independent variable (Factor 1) = time or condition
Time/condition 1 Time/condition 2 Time/condition 3
Group 1 Dependent variable Dependent variable Dependent variable
Group 2 Dependent variable Dependent variable Dependent variable
*
*
*
*
*
*
*

A mixed factor ANOVA is another omnibus test that is used to test 3 null hypotheses:
3. There is no signif icant within-subject effect, i.e. no significant difference between
the means of the differences between all the conditions/times.
4. There is no significant between-subject effect, i.e. no significant difference between
the means of the groups.
5. There is no significant interaction effect , i.e. no significant group differences across
conditions/time.
ASSUMPTIONS
Like all other parametric tests, mixed factor ANOVA makes a series of assumptions that should
either be addressed in the research design or can the tested for.
The within-subjects factor should contain at least two related (repeated measures)
categorical groups (levels)
The  between-subjects factor should have  at least two categorical  independent
groups (levels).
The dependent variable should be continuous and approximately normally distributed
for all combinations of factors.
There should be homogeneity of variance for each of the groups , and, if more than 2
levels) sphericity between the related groups.
There should be no significant outliers.

# MIXED FACTOR ANOVA (page 141)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_MIXED_FACTOR_ANOVA_PAGE_141]

# RUNNING THE MIXED FACTOR ANOVA (page 142)


Open 2-way Mixed ANOVA.csv in JASP. This contains 4 columns of data relating to the type
of weightlifting grip and speed of the lift at 3 different loads (%1RM). Column 1 contains the
grip type, and columns 2 -4 contain the 3 repeated measures (30, 50 and  70%). Check for
significant outliers using boxplots, then go to ANOVA > Repeated measures ANOVA.
Define the Repeated Measures Factor, %1RM, and add 3 levels (30, 50 and 70%). Add the
appropriate variable to the Repeated measures Cells and add Grip to the Between-Subjects
Factors:

Additionally, tick Descriptive statistics and Estimates of effect size (2).
In Descriptive plots, move %1RM to the horizontal axis and Grip to separate lines.  It is now
possible to add a title for the vertical axis.

# RUNNING THE MIXED FACTOR ANOVA (page 142)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_MIXED_FACTOR_ANOVA_PAGE_142]

# UNDERSTANDING THE OUTPUT (page 144)


The output should initially comprise 3 tables and 1 graph.
For the main effect for %1RM, t he within-subjects effects table reports a large F -statistic,
which is highly significant (p<.001) and has a large effect size (0.744). Therefore, irrespective
of grip type, there is a significant difference between the three %1RM loads.
However, JASP has reported under the table that the assumption of sphericity has been
violated. This will be addressed in the next section.
Finally, there is a significant interaction between %1RM and grip ( p<.001), which also has a
large effect size (0.499). This suggests that the differences between the %1RM loads are
affected somehow by the type of grip used.
For the main effect of grip, the between-subjects table shows a significant difference between
grips (p< .001), irrespective of %1RM.
From the descriptive data and the plot, it appears that there is a larger difference between
the two grips at the high 70% RM load.

# UNDERSTANDING THE OUTPUT (page 144)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_144]

# TESTING ASSUMPTIONS (page 145)


In Assumptions Checks, tick Sphericity tests, Sphericity corrections and Homogeneity tests.
Mauchlys test of sphericity is significant , so that assumption has been violated ; therefore,
the Greenhouse -Geisser correction should be used since epsilon is <0.75. Go back to
Assumption Checks and in Sphericity corrections, leave Greenhouse-Geisser only ticked.  This
will result in an updated Within-Subjects Effects table:

Levenes test shows that there is no difference in variance in the dependent variable between
the two grip types.
However, if the ANOVA reports no significant difference, you can go no
further in the analysis.

# TESTING ASSUMPTIONS (page 145)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_TESTING_ASSUMPTIONS_PAGE_145]

# POST HOC TESTING (page 146)


If the ANOVA is significant , post hoc testing can now be carried out. In Post Hoc Tests , add
%1RM to the analysis box on the right, tick Effect size, and, in this case, use Holm for the post
hoc correction. Only Bonferroni or Holms corrections are available for repeated measures.

The post hoc tests show that, irrespective of grip type, each load is significantly different from each of
the other loads, and as seen from the plot, lift velocity significantly decreases as the load increases.
Finally, in Simple main effects , add Grip to the Simple effect factor and %1RM to the
Moderator factor 1.

These results show that there is a significant difference in lift speed between the two grips at
30% 1RM and the higher 70% 1RM loads (p=0.035 and p<0.001, respectively).

# POST HOC TESTING (page 146)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_POST_HOC_TESTING_PAGE_146]

# REPORTING THE RESULTS (page 148)


Using the Greenhouse-Geisser correction, there was a significant main effect of load: F(1.48,
26.64) = 115.45, p<.001. Bonferroni corrected post hoc testing showed that there was a
significant sequential decline in lift speed from 30 -50% 1RM ( p=.035) and 50 -70% 1RM
(p<.001).
There was a significant main effect for grip type: F( 1, 18) = 20.925, p<.001, showing an overall
higher lift speed using the traditional rather than the reverse grip.
Using the Greenhouse-Geisser correction, there was a significant %1RM x Grip interaction: F
(1.48, 26.64) = 12.00, p<.001) showing that the type of grip affected lift velocity over the
%1RM loads.

# REPORTING THE RESULTS (page 148)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_REPORTING_THE_RESULTS_PAGE_148]

# CHI-SQUARE TEST FOR ASSOCIATION (page 149)


The chi-square (2) test for independence (also known as Pearson's 2 test or the 2 test of association)
can be used to determine if a relationship exists between two or more categorical variables. The test
produces a contingency table, or cross-tabulation, which displays the cross-grouping of the categorical
variables.
The 2 test checks the null hypothesis that there is no association between two categorical variables.
It compares the observed frequencies of the data with frequencies that would be expected if there
were no association between the two variables.
The analysis requires two assumptions to be met:
1. The two variables must be categorical data (nominal or ordinal)
2. Each variable should comprise two or more independent categorical groups
Most statistical tests fit a model to the observed data with a null hypothesis that there is no difference
between the observed and modelled (expected) data.  The error or deviation of the model is calculated
as:
Deviation =  (observed model) 2
Most parametric models are based on population means and standard deviations. The 2 model ,
however, is based on expected frequencies.
How are the expected frequencies calculated?  For example, we categorised 100 people into male,
female, short and tall. If there was an equal distribution between the 4 categories expected frequency
= 100/4 or 25% but the actual observed data does not have an equal frequency distribution.
Equal
Distribution
Male Female Row
Total
Tall 25 25 50
Short 25 25 50
Column Total 50 50
The model based on expected values can be calculated by:
Model (expected)  = (row total x column total)/100
Model  tall male   = (81 x 71) /100 = 57.5
Model  tall female   = (81 x 29) /100 = 23.5
Model  small male   = (19 x 71) /100 = 13.5
Model  small female   = (19 x 29) /100 = 5.5
These values can then be added to the contingency table:
Observed
Distribution
Male Female Row
Total
Tall 57 24 81
Short 14 5 19
Column Total 71 29

Male (M) Female (F) Row Total
Tall (T) 57 24 81
Expected 57.5 23.5
Short (S) 14 5 19
Expected 13.5 5.5
Column Total 71 29
The 2 statistic is derived from
(observed expected)
expected
2
Validity
2 tests are only valid when you have a reasonable sample size, that is, less than 20% of cells have an
expected count of less than 5, and none have an expected count of less than 1.

# CHI-SQUARE TEST FOR ASSOCIATION (page 149)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_CHI_SQUARE_TEST_FOR_ASSOCIATION_PAGE_149]

# RUNNING THE ANALYSIS (page 150)


The dataset Titanic survival is a classic dataset used for machine learning and contains data on 1309
passengers and crew who were on board the Titanic when it sank in 1912. We can use this to look at
associations between survival and other factors.  The dependent variable is Surviv al, and possible
independent variables are all the other values.

By convention, the independent variable is usually placed in the contingency table columns , and the
dependent variable is placed in the rows.
Open Titanic survival chi square.csv in JASP, in the spreadsheet tab and double click on Survived. This
will open up the Label editor, where you can add labels to the code. Survival has been coded as no =
0 and yes = 1.
Go to Frequencies > Contingency tables and add survived to rows as the dependent variable and sex
into columns as the independent variable.

To make the tables easier to read, in Preferences, I suggest that you reduce the number of
decimals to 1.
In statistics, tick all the following options:
In Cells, tick the following:

# RUNNING THE ANALYSIS (page 150)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_ANALYSIS_PAGE_150]

# UNDERSTANDING THE OUTPUT (page 153)


First, look at the Contingency table output.
Remember that 2 tests are only valid when you have a reasonable sample size, i.e. less than 20% of
cells have an expected count of less than 5, and none have an expected count of less than 1.
From this table, looking at % within rows, more males died on the Titanic compared to females , and
more females survived compared to males. But is there a significant association between gender and
survival?
The statistical results are shown below:
The 2 statistic (2 (1) = 365.9, p<.001) suggests that there is a significant association between gender
and survival.
2 continuity correction can be used to prevent the overestimation of statistical significance for small
datasets. This is mainly used when at least one cell of the table has an expected count smaller than 5.

As a note of caution , this correction may overcorrect and result in an overly conservative result that
fails to reject the null hypothesis when it should (a type II error).
The likelihood ratio is an alternative to the Pearson chi-square. It is based on the maximum-likelihood
theory. For large samples, it is identical to Pearson 2. It is recommended for small sample sizes, i.e.
<30.
Nominal measures, Phi (2 x 2 contingency tables only) and Cramer's V (most popular) are both tests
of the strength of association (i.e. effect sizes). Both values are in the range of 0 (no association) to 1
(complete association). The strength of association between the variables has a large effect size.
The Contingency coefficient is an adjusted Phi value and is only suggested for large contingency tables,
such as 5 x 5 tables or larger.
Effect size 4 df Small Moderate Large
Phi and Cramers V (2x2 only) 1 0.1 0.3 0.5
Cramers V 2 0.07 0.21 0.35
Cramers V 3 0.06 0.17 0.29
Cramers V 4 0.05 0.15 0.25
Cramers V 5 0.04 0.13 0.22
JASP also provides the Odds ratio (OR), which is used to compare the relative odds of the occurrence
of the outcome of interest (survival), given exposure to the variable of interest (in this case, gender).
In Preferences, I suggest that you return the number of decimals to 3.
For ease of understanding, take the reciprocal of the OR (1/0.088), i.e. 11.36. This suggests that male
passengers had 11.36 times more chance of dying than females.
4 Kim HY.  Statistical notes for clinical researchers: Chi-squared test and Fisher's exact test. Restor. Dent.
Endod. 2017; 42(2):152-155.

How is this calculated?  Use the counts from the contingency table in the following:
Odds[males]  = Died/Survived  =  682/162 = 4.209
Odds[females]   = Died/Survived  = 127/339 = 0.374
OR = Odds[males] / Odds [females]  = 11.3

# UNDERSTANDING THE OUTPUT (page 153)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_153]

# GOING ONE STEP FURTHER. (page 155)


We can also further decompose the contingency table as a form of post hoc testing by converting the
counts and expected counts in each cell to a standardised residual. This can tell us if the observed
counts and expected counts are significantly different in each cell.
The Pearson's residual for a cell in a table is a version of the standard z-score, calculated as
z = observed  expected
expected
The resulting value of z is then given a positive sign if observed>expected and a negative sign if
observed<expected. Z-score significances are shown below.
z-score P-value
<-1.96 or > 1.96 <0.05
<-2.58 or > 2.58 <0.01
<-3.29 or > 3.29 <0.001

When Pearsons residual z-scores are calculated for each cell in the contingency table, we can see
that the z-scores are significant in all cells.

# GOING ONE STEP FURTHER. (page 155)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_GOING_ONE_STEP_FURTHER_PAGE_155]

# UNDERSTANDING THE OUTPUT (page 156)


There is a significant association between gender and survival (2 (1) = 365.9, p<.001). The odds ratio
suggests that male passengers had 11.36 times more chance of dying than females.
Pearson's residuals suggest that significantly fewer women died than expected and significantly
more males died than expected.

# UNDERSTANDING THE OUTPUT (page 156)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_UNDERSTANDING_THE_OUTPUT_PAGE_156]

# META-ANALYSIS IN JASP (page 157)


Background
A meta-analysis is a statistical analysis that integrates results from multiple studies, providing a single
numerical value of the overall treatment effect for that group of studies. The difference between
statistical data analysis and meta -analysis is sho wn below. Effectively , each study becomes a
participant in the meta-analysis.
Statistical analysis Meta-analysis
Participant 1 Individual data Study 1 Study data
Participant 2 Individual data Study 2 Study data
Participant 3 Individual data Study 3 Study data
Participant 4 Individual data Study 4 Study data
Participant 5 Individual data Study 5 Study data
Overall group data & statistics Overall study group data & statistics
In JASP, click on the blue cross in the top right corner, and then tick the Meta-Analysis module. This
will add meta-analysis to the main ribbon.
Effect size and calculations
To perform a Meta -analysis in JASP , the overall effect size (ES) and standard error (SE) of the study
need to be provided.  An ES is a dimensionless estimate (no units) that indicates both the direction
and magnitude of the effect/outcome. The standard error measures the dispersion of different sample
means taken from the same population and can be estimated from a single sample standard deviation.
Some studies will provide this information, although many do not.  However, they will provide results
such as:
Central tendencies and dispersion
T or F statistics
P-values
Correlation coefficients
Chi-square
All of these can be converted to an ES and the SE determined using JASP's Effect size computation ,
which is described later.
For example, a study comparing a treatment to a control group may provide the variable mean, SD
and n of each group. From this , the ES, a standardised mean difference (d) , and estimated standard
error can be calculated.
To interpret the meta -analysis output, one needs to understand the following concepts :
heterogeneity, the model, effect size, and the forest plot.
Heterogeneity:
Heterogeneity describes any variability that may exist between the different studies.   It is the opposite
of homogeneity, which means that the population/data/results are the same.  There are 3 types:

Clinical:   Differences in participants, interventions, or outcomes
Methodological:  Differences in study design, risk of bias
Statistical:   Variation in intervention effects or results
If there is no variability, then the data can be described as homogeneous.  Meta-analysis is concerned
with statistical heterogeneity. Statistical heterogeneity is used to describe variability among
data/studies and occurs when the treatment effect estimates of a set of data/studies vary among one
another.  Studies with methodological flaws and small studies may overestimate treatment effects
and can contribute to statistical heterogeneity.
The diagram above5 shows examples of forest plots exhibiting low and high heterogeneity. In the low
condition, all the studies are generally lined up to the right of the vertical axis and all confidence
intervals are overlapping. In the high condition, the studies are spread over either side of the vertical
decision line and there is little overlap of the confidence intervals. Apart from visual observation of
Meta-analysis provides quantitative statistical methods to measure heterogeneity.
Tests for heterogeneity
Q Cochran Q test - values are given for 2 tests outlined below, (these tests have low statistical
power when only a few studies are included in the meta-analysis):
Test for residual heterogeneity tests the null hypothesis that all the effect sizes in the studies
are equal (homogeneity of effect sizes).
2     Tau square is an estimate of the total amount of heterogeneity. This is interpreted as
systematic, unexplained differences between the observed effects of the separate studies. It
is not affected by the number of studies, but it is often hard to interpret how relevant the
value is from a practical standpoint.
I2  This measures the extent of heterogeneity not caused by sampling error. If there is high
heterogeneity, a possible subgroup analysis could be done. If the value is very low, there is no
point in doing further subgroup analyses. I 2  is not sensitive to changes in the number of
studies and is therefore used extensively in medical and psychological research, especially
since there is a rule of thumb to interpret it.  A rough guide for interpretations has been
suggested as follows6:
0% to 40%: might not be important.
30% to 60%: may represent moderate heterogeneity.
50% to 90%: may represent substantial heterogeneity.
75% to 100%: considerable heterogeneity.
5 https://s4be.cochrane.org/blog/2018/11/29/what-is-heterogeneity/
6 Cochrane Handbook for Systematic Reviews of Interventions
Low
heterogeneity
High
heterogeneity

H2  The between-study variance is determined by equating the Q statistic to its expected value.
H2 has a value of 1 in the case of homogeneity , and heterogeneity is assumed to be present
when H2>1.
Meta-analysis models
There are two models commonly used in meta -analysis, and each make s different assumptions
relating to the observed differences among the studies.
Fixed effects model : this assumes that all studies share a common true ES , i.e. the data is
homogeneous.  All factors that could influence the ES are the same in all the study samples , and
therefore very little heterogeneity. Between-study differences are assumed to be due to chance and
thus not incorporated into the model. Therefore, each study included in the meta -analysis estimates
the same population treatment effect, which, in theory, represents the true population treatment
effect. Each study is weighted where more weight is given to studies with large sample sizes, i.e. more
information.
This model answers the following question: What is the best estimate of the population effect size?
Random effects model: This assumes a distribution of the treatment effect for some populations. i.e.
the different studies are estimating different, yet related, intervention effects. Therefore,
heterogeneity cannot be explained because it is due to chance.  This model assigns a  more balanced
weighting between studies.
This model answers the question What is the average treatment effect?.
It is therefore important to check for significant heterogeneity to
select the correct model for the meta-analysis.
Random effects model selection.
JASP provides 8 random effects estimators to estimate the 4 indices of heterogeneity (see tests for
heterogeneity above). All use a slightly different approach , resulting in different pooled ES
estimates and CIs. To date, the most often used estimator in medical research is the
DerSimonian-Laird estimator ; however, recent studies have shown better estimates of
between-study variance using Maximum -Likelihood, Restricted ML  (Default in JASP ),
Empirical Bayes and Sidak-Jonkman methods.
The Forest Plot.
This plot provides a visual summary of the meta-analysis findings. Graphically, it represents the ES and
95% CI for each study and an estimate of the overall ES and 95% CI for all the studies that were
included.  As previously mentioned, it can be used to visually assess the level of heterogeneity.

As a rule of thumb, if all the individual study CIs cover the final combined ES and its 95% CI (diamond),
then there is low heterogeneity (or high homogeneity). In this case, 6 studies do not intersect with the
diamond.
Funnel plot
Looking like an inverted funnel, this is a scatter plot of the intervention effect estimates from individual
studies against each studys size or precision. The intervention effect should increase as the size of the
study increases. Studies with low power will cluster at the bottom, while higher power studies will be
near the top. Ideally, the data points should have a symmetrical distribution.  Funnel plot asymmetry
indicates that the meta-analysis results may be biased. Potential reasons for bias include:
Publication bias (the tendency of authors/publishers to only publish studies with significant
results)
True heterogeneity
Artefact (wrong choice of effect measure)
Data irregularities (methodological design, data analysis, etc.)
Funnel plot asymmetry can be analysed statistically using either meta-regression, weighted regression
or rank correlation.

# META-ANALYSIS IN JASP (page 157)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_META_ANALYSIS_IN_JASP_PAGE_157]

# EFFECT SIZE COMPUTATION IN JASP (page 162)


Before running a meta-analysis, you will have to calculate the effect sizes and standard errors based
on the data extracted from the studies.
Open the Meta-Analysis fatigue.csv dataset. This is from papers comparing the effects of a new
treatment on the force recovery from eccentric exercise.  The data comprises of the means, SD and
N for control and treatment groups as well as the authors and publication date.
Click on the Meta-Analysis icon (this module may need to be added by clicking the blue plus sign).
Then click on Effect size computation.
This allows you to calculate effect sizes and standard errors from a variety of research designs and
measurements. In this case, we can use the default settings. Next, populate the appropriate cells:
You will see in the results that JASP has computed 21 effect sizes.

If you return to the spreadsheet viewer, you will see that 3 extra columns have been added:
Computed effect size
Computed standard error
Computed effect size type  in this case, the standardised mean difference (SMD)
These are used to run the meta-analysis.

# EFFECT SIZE COMPUTATION IN JASP (page 162)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_EFFECT_SIZE_COMPUTATION_IN_JASP_PAGE_162]

# RUNNING THE META-ANALYSIS IN JASP (page 163)


Go to Meta-Analysis and click on Classical Meta-Analysis. Add the computed effect size, computed
standard error and finally the Author to the Study Labels. Keep the method as Restricted ML.
In the Statistics tab, tick all 4 Heterogeneity measures, , 2, I2, H2.
This will result in 3 tables.

The Residual Heterogeneity test confirms significant heterogeneity in the data and that the
random-effects model (Restricted ML) is appropriate in this case. If this w ere not significant,
the method should be changed to fixed effects.
The Pooled Effect Size tests the H0 that all the estimates are zero, i.e. the interventions have
no significant effect. As can be seen, this is significant and H 0 can be rejected, thus the
intervention has a significant effect with an effect size difference of -0.352.
Looking at the Meta-Analytic Estimates, both H2 and I2 show excess variance (heterogeneity)
between the studies, thus supporting the use of a random-effects model.
Forest Plots
Return to the Statistics options and Forest Plot, tick Study information, and add both Author
and Date to the selected variables. Scroll down and tick Model information.

The Forest plot shows the weighted effect sizes (the size of the squares reflects the weight of
each study) and CIs used to determine the overall pooled ES (diamond).
The model information is just a summary of the first 3 tables above.
Funnel Plot
From the Meta -Analysis icon , click on Funnel Plot.  Add the computed effect size and
computed standard error to their relevant boxes. Also , click on Funnel under H1 and click
Funnel plot asymmetry tests.

The funnel plot shows that the observed effect sizes appear to be symmetrically distributed
around the vertical axis (based on the pooled effect size estimate, in this case, -0.352) and lie
within the 95% confidence triangle.  Asymmetry is often reported as being indicative of
publication bias.  This plot is accompanied by the  Meta-Regression Test for funnel plot
asymmetry, which in this case is non-significant (p=.387).
Reporting the results.
A random-effects model (Restricted ML) was used in the analysis. The Residual Heterogeneity
test confirms significant heterogeneity in the data  Q(20) = 49.4, p<.001.  This was supported
with an I 2 of 59.3%. The Pooled Effect Size  test showed a significant difference of -0.352
between the control and treatment studies, t(20) = -7.9, p<.001. Meta-regression showed no
Funnel plot asymmetry, Z=-0.865, p=0.387.
Also, show the Forest plot.

# RUNNING THE META-ANALYSIS IN JASP (page 163)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_RUNNING_THE_META_ANALYSIS_IN_JASP_PAGE_163]

# EXPERIMENTAL DESIGN AND DATA LAYOUT IN EXCEL FOR JASP IMPORT. (page 168)


Independent t-test
Design example:
Independent variable Group  1 Group 2
Dependent variable Data Data
Independent variable                                       Dependent variable
Categorical    Continuous
More dependent variables can be added if required

Paired samples t-test
Design example:
Independent variable Pre-test Post-test
Participant Dependent variable
1 Data Data
2 Data Data
3 Data Data
..n Data Data
Pre-test    Post-test

Correlation
Design example:
Simple correlation
Participant Variable 1 Variable 2 Variable 3 Variable 4 Variable ..n
1 Data Data Data Data Data
2 Data Data Data Data Data
3 Data Data Data Data Data
...n Data Data Data Data Data
Multiple correlations

Regression
Design example:
Simple Regression
Participant Outcome Predictor 1 Predictor 2 Predictor 3 Predictor ..n
1 Data Data Data Data Data
2 Data Data Data Data Data
3 Data Data Data Data Data
...n Data Data Data Data Data
Multiple regression

Logistic Regression
Design example:
Dependent Variable
(categorical)
Factor
(categorical)
Covariate
(continuous)
Participant Outcome Predictor 1 Predictor 2
1 Data Data Data
2 Data Data Data
3 Data Data Data
...n Data Data Data
More factors and covariates can be added if required

One-way Independent ANOVA
Design example:
Independent variable Group  1 Group 2 Group 3 Group...n
Dependent variable Data Data Data Data
Independent variable             Dependent variable
(Categorical)    (Continuous)
More dependent variables can be added if required

One-way repeated measures ANOVA
Design example:
Independent variable (Factor)
Participant Level 1 Level 2 Level 3 Level..n
1 Data Data Data Data
2 Data Data Data Data
3 Data Data Data Data
4 Data Data Data Data
..n Data Data Data Data
Factor (time)
Levels
(Related groups)
More levels can be added if required

Two-way Independent ANOVA
Design example:
Factor 1 Supplement 1 Supplement 2
Factor 2 Dose 1 Dose 2 Dose 3 Dose 1 Dose 2 Dose 3
Dependent
variable Data Data Data Data Data Data
Factor 1     Factor 2      Dependent variable
More factors and dependent variables can be added if required

Two-way Repeated measures ANOVA
Design example:
Factor 1
Interventions
Level 1
i.e. intervention 1
Level 2
i.e. intervention 2
Factor 2
Time
Level 1
i.e time 1
Level  2
i.e time 2
Level  3
i.e time 3
Level 1
i.e time 1
Level  2
i.e time 2
Level  3
i.e time 3
1 Data Data Data Data Data Data
2 Data Data Data Data Data Data
3 Data Data Data Data Data Data
..n Data Data Data Data Data Data
Factor 1 levels 1-n        Factor 2 levels 1-n

Two-way Mixed Factor ANOVA
Design example:
Factor 1
(Between subjects)
Group 1 Group 2
Factor 2 levels
(Repeated measures)
Trial 1 Trial 2 Trial 3 Trial 1 Trial 2 Trial 3
1 Data Data Data Data Data Data
2 Data Data Data Data Data Data
3 Data Data Data Data Data Data
..n Data Data Data Data Data Data
Factor 1    Factor 2 levels
(Categorical)                (Continuous)

Chi-squared - Contingency tables
Design example:
Participant Response 1 Response  2 Response 3 Response...n
1 Data Data Data Data
2 Data Data Data Data
3 Data Data Data Data
..n Data Data Data Data
All data should be categorical

# EXPERIMENTAL DESIGN AND DATA LAYOUT IN EXCEL FOR JASP IMPORT. (page 168)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_EXPERIMENTAL_DESIGN_AND_DATA_LAYOUT_IN_EXCEL_FOR_JASP_IMPORT_PAGE_168]

# SOME CONCEPTS IN FREQUENTIST STATISTICS (page 179)


The frequentist approach is the most commonly taught and used statistical methodology. It
describes sample data based on the frequency or proportion of the data from repeated
studies through which the probability of events is defined.
Frequentist statistics uses rigid frameworks including hypothesis testing, p values confidence
intervals etc.
Hypothesis testing
A hypothesis can be defined as  a supposition or proposed explanation made based on
limited evidence as a starting point for further investigation.
There are two simple types of hypotheses, a null hypothesis (H 0) and an alternative or
experimental hypothesis (H1). The null hypothesis is the default position for most statistical
analyses in which it is stated that there is no relationship or difference between groups.  The
alternative hypothesis states that there is a relationship or difference between groups in a
direction of difference/relationship. For example, if a study was carried out to look at the
effects of a supplement on sprint time in one group of participants compared to the placebo
group:
H0 = there is no difference in sprint times between the two groups
H1 = there is a difference in sprint times between the two groups
H2 = Group 1 is greater than Group 2
H3 = Group 1 is less than Group 2
Hypothesis testing refers to the strictly predefined procedures used to accept or reject the
hypotheses and the probability that this could be purely by chance . The confidence at which
a null hypothesis is accepted or rejected is called the level of significance. The level of
significance is denoted by , usually 0.05 (5%). This is the level of probability of accepting an
effect as true (95%) and that there is only 5% of the result is purely by chance.
Different types of hypotheses can easily be selected in JASP, however, the null hypothesis is
always the default.

Type I and II errors
The probability of rejecting the null hypothesis, when it is , in fact, true, is called Type I error
whereas the probability of accepting the null hypothesis  when it is not true  is called Type II
error.
The truth
Not guilty (H0) Guilty (H1)
The verdict
Guilty (H1)
Type I error
An i nnocent person
goes to prison
Correct decision
Not guilty (H0)
Correct decision
Type II error
A guilty person goes free
Type I error is deemed the worst error to make in statistical analyses.
Statistical power is defined as the probability that the test will reject the null hypothesis when
the alternative hypothesis is true. For a set level of significance, if the sample size  increases,
the probability of Type II error decreases, which therefore increases the statistical power.
Testing the hypothesis
The essence of hypothesis testing is to first define the null (or alternative) hypothesis , set
the criterion level , usually 0.05 (5%), collect and analyse sample data. Use a test statistic to
determine how far (or the number of standard deviations) the sample mean is from the
population mean stated in the null hypothesis. The test statistic is then compared to a critical
value. This is a cut-off value defining the boundary where less than 5% of the sample means
can be obtained if the null hypothesis is true.
If the probability of obtaining a difference between the means by chance is less than 5% when
the null hypothesis has been proposed, the null hypothesis is rejected and the alternative
hypothesis can be accepted.
The p-value is the probability of obtaining a sample outcome, given that the  value stated in
the null hypothesis is true. If the p-value is less than 5% (p < .05)  the null hypothesis  is
rejected.  When the p-value is greater than 5% (p > .05), we accept the null hypothesis.
Effect size
An effect size is a standard measure that can be calculated from any number of statistical
analyses. If the null hypothesis is rejected the result is significant.  This significance only
evaluates the probability of obtaining the sample outcome by chance but does not indicate
how big a difference (practical significance) is, nor can it be used to compare across different
studies.
The effect size indicates the magnitude of the difference between the groups. So for example,
if there was a significant decrease in 100m sprint times in a supplement compared to a

placebo group, the effect size would indicate how much more effective the intervention was.
Some common effect sizes are shown below.
Test Measure Trivial Small Medium Large
Between means Cohens d <0.2 0.2 0.5 0.8
Correlation Correlation coefficient (r)
Rank -biserial (rB)
Spearmans rho
<0.1
<0.1
<0.1
0.1
0.1
0.1
0.3
0.3
0.3
0.5
0.5
0.5
Multiple Regression Multiple correlation
coefficient (R)
<0.10 0.1 0.3 0.5
ANOVA Eta
Partial Eta
Omega squared
<0.1
<0.01
<0.01
0.1
0.01
0.01
0.25
0.06
0.06
0.37
0.14
0.14
Chi-squared Phi (2x2 tables only)
Cramers V
Odds ratio (2x2 tables only)
<0.1
<0.1
<1.5
0.1
0.1
1.5
0.3
0.3
3.5
0.5
0.5
9.0
In small datasets, there may be a moderate to large effect size but no significant differences.
This could suggest that the analysis lacked statistical power and that increasing the number
of data points may show a significant outcome. Conversely, when using large datasets,
significant testing can be misleading since small or trivial effects may produce statistically
significant results.
PARAMETRIC vs NON-PARAMETRIC TESTING
Most research collects information from a sample of the population of interest, it is normally
impossible to collect data from the whole population. We do, however, want to see how well
the collected data reflects the population in terms of the population mean, standard
deviations, proportions etc. based on parametric distribution functions.  These measures are
the population parameters. Parameter estimates of these in the sample population are
statistics. Parametric statistics require assumptions to be made of the data including the
normality of distribution and homogeneity of variance.
In some cases, these assumptions may be violated in that the data may be noticeably skewed:

Sometimes transforming the data can rectify this but not always. It is also common to collect
ordinal data (i.e. Likert scale ratings) for which terms such as mean and standard deviation
are meaningless. As such there are no parameters associated with ordi nal (non-parametric)
data. The non-parametric counterparts include median values and quartiles.
In both of the cases described non -parametric statistical tests are available. There are
equivalents to the most common classical parametric tests. These tests dont assume
normally distributed data  or population parameters and are based on sorting the data into
ranks from lowest to highest values. All subsequent calculations are done with these ranks
rather than with the actual data values.

# SOME CONCEPTS IN FREQUENTIST STATISTICS (page 179)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_SOME_CONCEPTS_IN_FREQUENTIST_STATISTICS_PAGE_179]

# WHICH TEST SHOULD I USE? (page 183)


Comparing one sample to a known or hypothesized population mean.
Testing relationships between two or more variables
Data type
Continuous
Ordinal
Nominal
Are parametric
assumptions met?
Yes
No
Pearsons
correlation
Spearmans or
Kendalls tau
Chi-square
contingency
tables
Continuous
Ordinal
Nominal
2 categories
>2 categories
One-sample
t-test
One-sample
median test
Binomial test
Multinomial test
or Chi-square
goodness of fit
Data type
Currently not available in
JASP

Predicting outcomes
Testing for differences between two independent groups
Data type
Continuous
Ordinal
Nominal
More than one
predictor variable?
No
Yes
Simple
regression
Ordinal
regression
Logistic
regression
tables
Multiple
regression
Currently not available in
JASP
Data type
Continuous
Ordinal
Nominal
Are parametric
assumptions met?
Yes
No
Independent
t-test
Mann-Whitney U
test
Chi-square or
Fischers Exact
test

Testing for differences between two related groups
Testing for differences between three or more independent groups
Data type
Continuous
Ordinal
Nominal
Are parametric
assumptions met?
Yes
No
Paired samples t-test
Wilcoxons test
McNemars test
Currently not available in
JASP
Data type
Continuous
Ordinal
Nominal
Are parametric
assumptions met?
Yes
No
ANOVA
Kruskall-Wallis
Chi-square
Contingency tables

Testing for differences between three or more related groups
Test for interactions between 2 or more independent variables
Data type
Continuous
Ordinal
Nominal
Are parametric
assumptions met?
Yes
No
RMANOVA
Friedman test
Repeated measures
logistic regression
Data type
Continuous
Ordinal
Nominal
Are parametric
assumptions met?
Yes
No
Two-way
ANOVA
Ordered logistic
regression
Factorial logistic
regression

# WHICH TEST SHOULD I USE? (page 183)-IMAGE DESCRIPTIONS
[IMAGE_PLACEHOLDER_WHICH_TEST_SHOULD_I_USE_PAGE_183]