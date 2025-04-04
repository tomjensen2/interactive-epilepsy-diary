# Interactive Seizure Diary Visualisation
An html/Javascript Interactive epilepsy seizure diary (Python/Plotly.Dash version also)

Epilepsy is a medical condition affecting approximately 65 million people worldwide.  Of these people ~30% fail to be effectively treated by anti-seizure drugs.  Many of these patients will be switching between different drug combinations with the aim of better controlling their seizures whilst recording when they have their seizures using a seizure diary.

In the absence of effective wearable medical devices Epileptologists ultimately have to judge the efficacy of a change in medication from the patients seizure diary, with limited time in office a good method for visualisation may be valuable to pick out trends in changes in seizures over time. 

This code is an adaptation of https://github.com/tomjensen2/Epilepsy_Seizure_Diary_visualisation and uses html/javascript or plotly.dash to generate an interactive seizure diary from diary data stored as .csv or excel file formats.  It provides a clickable heatmap chart of seizure count vs hour that can be viewed with different binning periods.  And a comparison tool where two time periods can be easily compared.

<strong>The current updated version now works with any properly formatted file (a template excel can be downloaded with instructions).  And anything can be logged whether it be seizures or not. Additional tools are present to identify triggers, the event dependence tool allows you to select trigger variables around which the probability of other selected events can be calculated.

<strong>Click the heatmap to plot detailed data, double click to compare time periods</strong>

![image](https://github.com/user-attachments/files/19132181/Epilepsy.Seizure.Diary.Dashboard.pdf)

To Do’s

-- Better/more relevant statistics

-- Sleep/exercise

-- import of new data

:credit to chatGPT for help getting this done 
