# Neural Network Repo, RACECAR@MIT

The objective of this approach is to allow for some high-level localization of the car through monocular vision, e.g., being able to know in which hallway the car is just by processing an image.

## Classification

This approach seeks to uniquely tailor the network for racing in the basements of the Stata center. The map for this venue is:
![alt text][stata-map]

We sectioned the map into 11 distinct categories so that, dependent upon the area in which the car thinks it is, we can tailor the control parameters to improve our performance:
![alt text][sectioned-stata-map]

## Performance Metrics

Of course, our only performance metrics were:
  * Time
  * Coolness
  * Ability to drift

## Implementation

... 
























[stata-map]: https://github.com/tonioteran/racecar-mit-nn/blob/master/map/basement_hallways_5cm.png "Stata Tunnels Map" 
[sectioned-stata-map]: https://github.com/tonioteran/racecar-mit-nn/blob/master/map/sectionedStataMap.png "Stata Tunnels With Sections"

