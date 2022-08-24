# ASDI-Hackathon

## Abstract

'Today, 56% of the world's population - 4.4 billion inhabitants - live in cities. This trend is expected to continue. By 2050, with the urban population more than doubling its current size, nearly 7 of 10 people in the world will live in cities.'

The World Bank
https://www.worldbank.org/en/topic/urbandevelopment/overview#:~:text=Today%2C%20some%2056%25%20of%20the,world%20will%20live%20in%20cities

Green spaces improve both the environmental and social conditions of cities, including air quality, population satisfaction and urban temperatures. Multiple datasets available through the Amazon Sustainability Data Initiative can be leveraged to inform locations for green spaces that would maximise their benefits on these urban conditions. We have extracted the raw data from their associated S3 buckets, processed them by way of upsampling them to allow for a high enough resolution that would be make the dashboard actually useable for city planners. Current dashboards of the like fail to reach this resolution. The final dashboard aggregates the underlying data into a green space score. 

The calculation of the score has been left open-ended due to the inevitable intricacies of underlying mechanisms between the data and the limitations of the data. The air quality dataset, for example, was the 3rd attempt at extracting useable data of the like. SILAM was explored but it relied on physical sensors which were too few in number to provide the resolution we wanted, accurately. OpenAQ was also explored, but besides some data quality issues in the S3 bucket data, it again had too few physical sensors that could be upsampled to a high enough resolution. Sentinel-5P was satellite data which promised to provide a higher resolution from the get go given there was no reliance on physical sensors. Besides having to figure out which of the NetCDF files included coverage of our chosen city, London, the main issue was that the data was presented as the total vertical column of e.g. Nitrogen Dioxide. In other words, the number of molecules from a 2D planar view of the Earth in the atmosphere. While an inaccurate representation of near-surface Nitrogen Dioxide levels, it provides a somewhat useable approximation. It was unfortunate that standard Air Qaulity indices such as does produced by the Environmental Protection Agency (EPA) could then not be used to properly contextualise the values.

Overlays for each type of data should provide enough transparency for city planners to understand why is the green score that value at that location.

Make point resolution is itself a parameterised variable given the weighted KNN we have deployed

Functions for an Air Quality Score and Greenspace Score have been derived naively with the latter having at least some input from WHO standards. The naive formulas are only meant to serve an MVP product. We would welcome domain experts to provide their insight into calibrating these formulas to be more informed.

Vincenty not supported by SKLearn KNN algorithm so decided with going with Haversine given the convenience of it being supported. The error between them was so marginal, especially at our resolution that the benefit was limited anyway.

Xarray package recent release did not work in Sagamaker

