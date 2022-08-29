# ASDI-Hackathon: Greenspace Suggestion Dashboard

## Abstract
'Today, 56% of the world's population - 4.4 billion inhabitants - live in cities. This trend is expected to continue. By 2050, with the urban population more than doubling its current size, nearly 7 of 10 people in the world will live in cities.'

The World Bank
https://www.worldbank.org/en/topic/urbandevelopment/overview#:~:text=Today%2C%20some%2056%25%20of%20the,world%20will%20live%20in%20cities

Green spaces improve both the environmental and social conditions of cities. Air quality, population satisfaction, urban temperatures, biodiversity, flood risk reduction, noise noise abatement, these are among the major benefits of greenspaces on an urban area. Multiple datasets available through the Amazon Sustainability Data Initiative can be leveraged to inform locations for green spaces that would maximise their benefits on these urban conditions. We have extracted the raw data from their associated S3 buckets, processed them by way of upsampling them to allow for a high enough resolution that would make the dashboard useable by city planners. The resolution itself is a parameter, with the distance weighted KNN models able to handle a higher resolution should said city planner wish it. As far as we are aware, current dashboards of the like fail to reach this resolution with the breadth of data that we have provided. The final dashboard aggregates the underlying data into a green space score that summarises where potential greenspaces would most benefit the aforementioned urban conditions.

The calculation of the score has been left open-ended due to the inevitable intricacies of underlying mechanisms between the data and the limitations of the data. The air quality dataset, for example, was the 3rd attempt at extracting useable data of the like. SILAM was explored but it relied on physical sensors which were too few in number to provide the resolution we wanted, accurately. OpenAQ was also explored, but besides some data quality issues in the S3 bucket data, it again had too few physical sensors that could be upsampled to a high enough resolution. Sentinel-5P was satellite data which promised to provide a higher resolution from the get go given there was no reliance on physical sensors, however the coverage of London (our chosen demo city) was sporadic. Besides having to figure out which of the NetCDF files included coverage of London, the main issue was that the data was presented as the total vertical column of e.g. Nitrogen Dioxide. In other words, the number of molecules from a 2D planar view of the Earth in the atmosphere. While an inaccurate representation of near-surface Nitrogen Dioxide levels, it provides a somewhat useable approximation. It was unfortunate that standard Air Qaulity indices such as those produced by the Environmental Protection Agency (EPA) could then not be used to properly contextualise the values. Our Air Quality Score itself therefore was naive in its formulation. With further time, atmospheric models could be used to approximate near-surface levels better.

A cornerstone of our project was the weighted distance K-Nearest Neighbour models employed to upsample the datasets to our resolution. These models considered n number of neighbours and then aggregated their values while weighting how far away they are from the map coordinate value to be predicted. Our inputs were longitudes and latitudes so working out the distance required preprocessing to consider the speherical nature of the Earth that these coordinates reflect. We explored two methods for converting these distances into something useable by the KNN models, we wanted to collapse a 3D space into a 2D one that the models would understand. The Vincenty and Haversine formulae work in similar ways but consider different levels of nuance as far as the shape of the Earth is concerned. The Earth is not a perfect sphere, it is elliptical, with the equator being a little fatter than would otherwise be in a sphere. Vincenty considers the Earth's elliptical nature, while the Haversine is a little more naive in this way. At the resolution we were using, and at the scale of a single city, the benfits of using Vincenty over Haversine was marginal. It just so happened that SKLearn facilitated Haversine, so we chose to go with that.

Using the land type API calls from Ordinance Survey data that return a boolean if a certain land type exists at a certain lattiude and longitude, we have engineered a distance from the nearest greenspace feature that enables greater clairty on where greenspaces are needed.

## Sustainability Initiatives Met
In its current state, we meet the following sustainability intitiaves, then bullet points

Ask Liam

## Future Work
Besides obviously expanding the datasets and getting true domain experts to refine the greenspace calculation, we see an expansion of scope of this project to envelope other major cities of the world. Once this is achieved, further aggregated analysis can be provided on the dashboard such as relative metrics compared to other cities. We have also explored developing a data pipeline to facilitate new data through processing jobs on Sagemaker; this cloud infrastructuture would be integral to any expansion going forward, particularly given the option for scaleability as the data becomes larger and larger. Our resolution of 250m is a parameter, the KNN weighted distance models facilitate any resolution, hwoever, of course, ideally, one would prefer to add higher resolution data to begin with so that its predictions do not become too detracted from reality.

Review notebooks comments

We have demoed our dashboard via Plotly Dash, which suited our needs for a Minimum Viable Product, however there are major drawbacks with regards to the performance of plotting at this resolution; in fact we became limited to 250m given Plotly Dash started to crash. Given more time, we would implement our dashboard in AWS QuickSight given their affinity with large datasets, not to mention the easy integration available with other cloud infrastructure.

Overlays for each type of data should provide enough transparency for city planners to understand why is the green score that value at that location.

Make point resolution is itself a parameterised variable given the weighted KNN we have deployed

Functions for an Air Quality Score and Greenspace Score have been derived naively with the latter having at least some input from WHO standards. The naive formulas are only meant to serve an MVP product. We would welcome domain experts to provide their insight into calibrating these formulas to be more informed.

Vincenty not supported by SKLearn KNN algorithm so decided with going with Haversine given the convenience of it being supported. The error between them was so marginal, especially at our resolution that the benefit was limited anyway.

Xarray package recent release did not work in Sagemaker

Include images from EDA and final dashboard

Formulae?

![Image 1](/Dashboard_Images/Air_Quality_Score.png)

![Image 2](/Dashboard_Images/Greenspace_Score.png)

