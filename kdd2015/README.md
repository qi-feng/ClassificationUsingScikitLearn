# KDD Cup 2015 
We competed as team BoilerUp in KDD Cup 2015 and ranked 57 among 821 teams (https://www.kddcup2015.com/). The objective of the competition is to predict whether a user will drop out of a course after enrollment based on data provided by XuetangX, a Chinese Massive Open Online Course (MOOC) platform. 

# What are data like?
- enrollment_train.csv
  * Simplest file that we have, 120542 rows. 
  * columns: enrollment_id,username,course_id 
  * Course_id (total of 39 unique courses) should provide useful information. There are easy courses and hard ones. The difficulty should affect dropout rates and help our predictions. 
  * Username has 79186 unique values, on average one user only takes ~1.5 courses. Too many dimensions and seems not very helpful. 

- log_train.csv
  * The work horse for our binary classification problem. 8157277 lines. 
  * columns: enrollment_id,time,source,event,object
  * There are 7 different events: ['nagivate', 'access', 'problem', 'page_close', 'video', 'discussion', 'wiki']

- object.csv
  * columns: course_id,module_id,category,children,start
  * The start time offers valuable information. We can compare the start time of a module with the time that a given enrollment accesses this module. If an enrollment tends to access a module shortly (e.g. within a week) after the module starts, it should be likely that this enrollment will not dropout. 
  * Categories are: ['about', 'chapter', 'course', 'course_info', 'html', 'outlink',
       'problem', 'sequential', 'static_tab', 'vertical', 'video',
       'combinedopenended', 'peergrading', 'discussion', 'dictation']
  * Categories that has a start that is not 'null' are: ['chapter', 'course', 'sequential']
  

# What features? 
Roughly, we want to assess the difficulty of the course, the diligence and consistency of the user, and other non-trivial aspects that may affect the probability of an enrollment being successful (i.e. no dropout). 

- **Enrollment profile** (a user taking a course):
  * number of events
  * categorical number of events (7 categories)
  * important event types: access & navigate
      * what module the event is accessing/navigating to
      
  * time of each event:
      * time of the first/last access to problems and videos with respect to the course start time
      * duration of all activities
      * power spectrum of the series of events to access how regular/even the activities are distributed in time

- **User profile**: 
  * number of courses taken
  
- **Course profile** (not used): 
  * number of users 
  * number of modules 
  * number of chapters/videos/problem_sets 
  * dropout rate 

- **Relational**: 
  * ratio # of user events / # of course modules 
  * start time of user events - time of module starts 

The final features have 106 dimensions (with dummy variables e.g. 39 course ids count as 39 dimensions but really belong to one feature). 

# What models? 
* We have used four models: 
  * xgboost (https://github.com/dmlc/xgboost), 
  * gradient boosting in scikit-learn (http://scikit-learn.org/stable/modules/ensemble.html), 
  * calibrated random forest in scikit-learn (http://scikit-learn.org/stable/modules/ensemble.html), and 
  * neural network from lasagne (https://github.com/Lasagne/Lasagne) and nolearn (https://github.com/dnouri/nolearn). 
* These models were trained using randomly selected 90% of the traning data. The performance of each model is evaluated based on the area under the ROC curve (AUC; between 0 and 1, and the higher the better) calculated from the rest 10% of the training data. 
* Gradient boosting and xgboost both give the best single-model performence with AUC ~0.87+. Neural network seemed to give us the worst single-model performance among the above. 
* By combining multiple models, we were able to reach AUC ~0.891. The winning AUC is ~0.909. 
