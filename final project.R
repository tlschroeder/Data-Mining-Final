#Data mining final project
#Greg Gardner, Timothy Schroeder
library(DAAG)
library(glmnet)
library(plyr)
library(dplyr)
library(tm)
library(topicmodels)
library(MASS)
library(ggplot2)
library(nnet)
library(mgcv)
library(gamclass)

listings <- read.csv('listings.csv')

################################Data Cleaning#######################################

#strip irrelevant columns
#NOTE: id is retained to merge text data back in later, but should not be included in model.
listingsReduced <- listings[,c('id','host_since','host_response_time','host_response_rate','host_acceptance_rate',
                               'neighbourhood_cleansed', 'host_listings_count',
                               'host_has_profile_pic','host_identity_verified','property_type', 'room_type',
                               'accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','guests_included',
                               'extra_people','minimum_nights','maximum_nights','review_scores_rating',
                               'cancellation_policy', 'host_about', 'summary')]
#dropped fields are based on unparseable data (image urls), containing the response (review subscores/superuser), 
#containing other fields (multiple neighborhood/address fields), or irrelevance (host id)

#per background research, add profile length as a column
listingsReduced$hostLength <- unlist(lapply(listings$host_about, function(x) nchar(as.character(x))))

#convert dollars and percentages to integers
suppressWarnings({
listingsReduced$host_response_rate <- unlist(lapply(listings$host_response_rate, function(x) as.numeric(sub("%", "", x))))
listingsReduced$host_acceptance_rate <- unlist(lapply(listings$host_acceptance_rate, function(x) as.numeric(sub("%", "", x))))
})
listingsReduced$price <- unlist(lapply(listings$price, function(x) as.numeric(gsub("[^0-9]","",x))/100))
listingsReduced$extra_people <- unlist(lapply(listings$extra_people, function(x) as.numeric(gsub("[^0-9]","",x))/100))

#exclude entries with missing data
#NOTE ON EXCLUSIONS: Necessary exclusions included hosts with no summaries/profiles and no prior stays or reviews.
#Once those are controlled for, <10 observations had missing data.
listingsReduced <- na.omit(listingsReduced) 
listingsReduced <- listingsReduced[!(as.character(listingsReduced$host_about) == ''),]
listingsReduced <- listingsReduced[!(as.character(listingsReduced$summary) == ''),]
listingsReduced <- listingsReduced[unlist(lapply(as.character(listingsReduced$host_about), function(x) nchar(x) > 3)),]
listingsReduced <- listingsReduced[unlist(lapply(as.character(listingsReduced$summary), function(x) nchar(x) > 3)),]
corpusBase <- listingsReduced # <-------corpus gets built off this output

#drop unused fields
listingsReduced$summary <- NULL
listingsReduced$host_about <- NULL
listingsReduced$id <- NULL
#transform amenities from a list into boolean variables
#get clean lists from each row
amenities <- as.character(listingsReduced$amenities)
amenities <- lapply(amenities, function(x) strsplit(gsub('(["\\\\\\{\\}])','',x),split = ','))

#get list of unique values
values <- list()
for (i in (1:nrow(listingsReduced))){
  values <- c(values, unlist(amenities[[i]]))
}
values <- unique(values)

#remove invalid values
values[26:27] <- NULL

#add values as new columns
for (i in 1:length(values)){
  listingsReduced[,values[[i]]] <- NA
}

#populate new columns
for (i in (1:nrow(listingsReduced))){
  for(i2 in (1:length(values))){
    listingsReduced[i,values[[i2]]] <- values[[i2]] %in% unlist(amenities[i])
  }
}

#remove old unused version
listingsReduced$amenities <- NULL

#convert 'host_since' from a date to the number of weeks that an individual has been a host
listingsReduced$host_since <- lapply(as.POSIXct(listingsReduced$host_since, '%Y-%m-%d', tz = 'EST'), function(x) 
  as.numeric(difftime(as.POSIXct('2016-09-07', tz = 'EST'),x), unit = 'weeks'))
listingsReduced$host_since <- as.numeric(listingsReduced$host_since)

#convert variables to factor as appropriate
factors <- c('host_response_time','neighbourhood_cleansed','host_has_profile_pic',
             'host_identity_verified','property_type','room_type','bed_type','cancellation_policy')
listingsReduced[,factors] <- lapply(listingsReduced[,factors], as.factor)

#clean up factor levels
listingsReduced <- droplevels(listingsReduced)

#We want to also take a look at our response variable to prevent issues down the road:
quantile(listingsReduced$review_scores_rating)
#0%  25%  50%  75% 100% 
#20   90   95   98  100

#Pretty skewed. Let's do residual checks on a naive model.
simpleMod <- lm(review_scores_rating~., data = listingsReduced)
plot(simpleMod$fitted.values, simpleMod$residuals, main = 'Naive Linear Model Residuals')
qqnorm(simpleMod$residuals)
#adjust r-squared .169

#Also bad. Let's check and see if y transformation is a good idea or not:
bc <- boxcox(simpleMod)
bc
#data is nonlinear to the point of being off the map of a boxcox. Can still try linear models, but may need an alternate approach.

#So, build a bucketed response variable for logistic regression
sum(listingsReduced$review_scores_rating == 100) / nrow(listingsReduced)

#The most popular score was a perfect 100, comprising about a full fifth of the data set
#We want that mostly in its own 'bucket', so we decide to build quintiles.
buckets <- quantile(listingsReduced$review_scores_rating, probs = seq(0,1,.5))
#0%  50% 100% 
#20   95  100 

#literature suggests J-curve should start around 90%; guests on airbnb also RECEIVE reviews, so a higher start point makes sense
listingsBucketed <- listingsReduced
listingsBucketed$review_scores_rating <- with(listingsReduced, 
                                             cut(review_scores_rating, breaks=quantile(review_scores_rating, 
                                            probs=seq(0,1, by=0.5), na.rm=TRUE), include.lowest=TRUE))
levels(listingsBucketed$review_scores_rating) <- c('bad','good')
qplot(listingsBucketed$review_scores_rating)

#check a naive model for oddities
testglm <- glm(review_scores_rating~., data=listingsBucketed, family = 'binomial')
#it seems that the data is in some cases perfectly linearly separable; let's check and see why
library(safeBinaryRegression)
testglm <- glm(review_scores_rating~., data=listingsBucketed, family = 'binomial')

#version that deals with some name weirdness
listingsReducedClean <- listingsReduced
names(listingsReducedClean) <- as.character(gsub('`| |/|-|\\(|\\)|2|4','',names(listingsReducedClean)))

######################################Pre-Text Modeling (linear)#######################################
#Sticking with model selection techniques that can eliminate variables, since we have a whole bunch and
#we want an interpretable model

####stepwise model selection
s.null <- lm(review_scores_rating~1, data=listingsReduced)
s.full <- lm(review_scores_rating~., data=listingsReduced)
stepwiseSelection <- step(s.null, scope=list(lower=s.null, upper=s.full), direction="both")
forwardSelection <- step(s.null, scope=list(lower=s.null, upper=s.full), direction="forward") #identical to result of stepwise selection
backwardSelection <- step(s.full, scope=list(lower=s.null, upper=s.full), direction="backward")
#This part's kind of weird and needs an explanation:
#To avoid issues with unseen factor levels in different folds, dummy variables need to be used for
#cross validation. However, dummy variables CANNOT be used for stepwise selection because the stepwise
#algorithm doesn't 'know' to add or remove dummy variables as entire sets and not individually.
#So, we run stepwise selection on the normal dataframe and then break it up into dummy variables
#via model.matrix for cross-validation.

#turn stepwise selection model output into an indexing vector
keptVariables <- as.character(stepwiseSelection$call)[2]
keptVariables <- gsub('~','+',keptVariables)
keptVariables <- gsub('`','',keptVariables)
keptVariables <- strsplit(keptVariables,'\\+')
keptVariables <- as.vector(keptVariables)[[1]]
keptVariables <- trimws(keptVariables, 'both')

#get the dataframe with dummy variables
modelingMatrix <- model.matrix(review_scores_rating~., data = listingsReduced[,keptVariables])
modelingFrame <- as.data.frame(modelingMatrix)
modelingFrame$`(Intercept)` <- NULL
modelingFrame$review_scores_rating <- listingsReduced$review_scores_rating

#cross validate
stepwise.fit <- lm(review_scores_rating ~ ., data = modelingFrame) #use all the data because modelingFrame already has the reduced set
stepwisecv <- cv.lm(modelingFrame, stepwise.fit, m = 10)
#^Warnings about rank deficiency are likely due to small sample size; checked for bad dummy variables, and that
#isn't the case here. Definitely want to use vif and pull excess out of whatever final model.
#MSE: 61.5
#Adjusted R-squared: 0.19

###repeat for backwards selection model
keptVariables <- as.character(backwardSelection$call)[2]
keptVariables <- gsub('~','+',keptVariables)
keptVariables <- gsub('`','',keptVariables)
keptVariables <- strsplit(keptVariables,'\\+')
keptVariables <- as.vector(keptVariables)[[1]]
keptVariables <- trimws(keptVariables, 'both')

#get the dataframe with dummy variables
backModelingMatrix <- model.matrix(review_scores_rating~., data = listingsReduced[,keptVariables])
backModelingFrame <- as.data.frame(backModelingMatrix)
backModelingFrame$`(Intercept)` <- NULL
backModelingFrame$review_scores_rating <- listingsReduced$review_scores_rating

#cross validate
backward.fit <- lm(review_scores_rating ~ ., data = backModelingFrame) #use all the data because modelingFrame already has the reduced set
backwardcv <- cv.lm(backModelingFrame, backward.fit, m = 10)
#MSE: 60.4
#Adjusted R-squared: .176
#still has the warning regarding rank-deficient data.

###Repeat for forward selection model
keptVariables <- as.character(forwardSelection$call)[2]
keptVariables <- gsub('~','+',keptVariables)
keptVariables <- gsub('`','',keptVariables)
keptVariables <- strsplit(keptVariables,'\\+')
keptVariables <- as.vector(keptVariables)[[1]]
keptVariables <- trimws(keptVariables, 'both')

#get the dataframe with dummy variables
forwardModelingMatrix <- model.matrix(review_scores_rating~., data = listingsReduced[,keptVariables])
forwardModelingFrame <- as.data.frame(forwardModelingMatrix)
forwardModelingFrame$`(Intercept)` <- NULL
forwardModelingFrame$review_scores_rating <- listingsReduced$review_scores_rating

#cross validate
forward.fit <- lm(review_scores_rating ~ ., data = forwardModelingFrame) #use all the data because modelingFrame already has the reduced set
forwardcv <- cv.lm(forwardModelingFrame, forward.fit, m = 10)
#MSE: 61.5
#R-Squared: .176

####lasso regression
#create lambdas and x matrix
grid <- 10^seq(10, -10, length = 100)
x<- model.matrix(review_scores_rating~., data = listingsReduced)

#create model
lasso.mod <- glmnet(x,listingsReduced$review_scores_rating, alpha =1, lambda = grid, thresh = 1e-12)

#10-fold cross-validation
lasso.out <- cv.glmnet(x, listingsReduced$review_scores_rating, lambda = grid, alpha = 1)
#plot(lasso.out)

#find the best lambda
bestlam <- lasso.out$lambda.min

#predict using the best lambda
lasso.predicts <- predict(lasso.mod, s = bestlam, newx = x)

MSE<-mean((lasso.predicts-listingsReduced$review_scores_rating)^2) 
#65.72766 MSE
coef(lasso.out, id = which.min(lasso.out$lambda))

#R-squared
SSR.lasso <- sum((lasso.predicts- mean(listingsReduced$review_scores_rating))^2)
SST.lasso <- sum((listingsReduced$review_scores_rating - mean(listingsReduced$review_scores_rating))^2)
R.squared.lasso <- SSR.lasso/SST.lasso 
#.0171961 R-Squared

#Backwards model has a superior number of variables; we adopt it going forward

#Now that we have a 'best' model from this group (backwards selection) let's dig a little deeper.
vif(backward.fit)
#the host_response_time values have abnormally high vifs; given our earlier concerns over rank deficiency, we should probably drop them
#(And, no, it isn't an improperly-formed dummy variable)
backModelingFrame$`host_response_timewithin a day` <- NULL
backModelingFrame$`host_response_timewithin a few hours` <- NULL
backModelingFrame$`host_response_timewithin an hour` <- NULL

backward.fit <- lm(review_scores_rating ~ ., data = backModelingFrame)
backwardcv <- cv.lm(backModelingFrame, backward.fit, m = 10)
#MSE: 60.6
#adjusted r-squared: .166

#residuals plot!
plot(backward.fit$fitted.values, backward.fit$residuals)
qqnorm(backward.fit$residuals)
#Pretty bad. Let's check and see if y transformation is a good idea or not:
boxcox(backward.fit)


#####################################Pre-text Modeling (non-linear)#######################################

###### Ordinal Regression for Pre-Text Model ########

#Install packages
wants <- c("MASS", "ordinal", "rms", "VGAM")
has   <- wants %in% rownames(installed.packages())
if(any(!has)) install.packages(wants[!has])

#Create model
library(rms)

#Select variables
ord.null = lm(review_scores_rating ~ 1, data = listingsReduced)
ord.full = lm(review_scores_rating ~ ., data = listingsReduced)

#Forward selection
ord.f = step(ord.null, scope = list(lower = ord.null, upper = ord.full), direction = "forward")

ordf.reg = orm(formula = review_scores_rating ~ Essentials + neighbourhood_cleansed + 
                  Washer + host_listings_count + Kitchen + price + property_type + 
                  Doorman + host_identity_verified + host_response_time + host_response_rate + 
                  `Family/Kid Friendly` + `Air Conditioning` + `Hair Dryer` + 
                  cancellation_policy + `Fire Extinguisher` + guests_included + 
                  extra_people + bedrooms + accommodates + Shampoo + `Pets Allowed` + 
                  Heating, data = listingsReduced)


#McFadden Pseudo R-squared of the model
library(VGAM)
vglm0 = vglm(review_scores_rating ~ 1, family = propodds, data = listingsReduced)

vglmf.reg = vglm(formula = review_scores_rating ~ Essentials + neighbourhood_cleansed + 
                   Washer + host_listings_count + Kitchen + price + property_type + 
                   Doorman + host_identity_verified + host_response_time + host_response_rate + 
                   `Family/Kid Friendly` + `Air Conditioning` + `Hair Dryer` + 
                   cancellation_policy + `Fire Extinguisher` + guests_included + 
                   extra_people + bedrooms + accommodates + Shampoo + `Pets Allowed` + 
                   Heating, family = propodds, data = listingsReduced)

LLf = logLik(vglmf.reg)
LL0 = logLik(vglm0)

r2 = (1 - (LLf/LL0)) #Adjusted R^2 is 0.0403


#Backward selection
ord.b = step(ord.full, scope = list(lower = ord.null, upper = ord.full), direction = "backward")

ordb.reg = orm(formula = review_scores_rating ~ host_since + host_response_time + 
                 host_response_rate + host_acceptance_rate + neighbourhood_cleansed + 
                 host_listings_count + host_identity_verified + accommodates + 
                 bedrooms + price + guests_included + extra_people + cancellation_policy + 
                 `Air Conditioning` + Kitchen + `Pets Allowed` + Heating + 
                 `Family/Kid Friendly` + Washer + `Fire Extinguisher` + Essentials + 
                 Shampoo + `Hair Dryer` + `Cat(s)` + Doorman, data = listingsReduced)

#McFadden Pseudo R-squared of the model
vglm0 = vglm(review_scores_rating ~ 1, family = propodds, data = listingsReduced)

vglmb.reg = vglm(formula = review_scores_rating ~ host_response_time + host_response_rate + 
                    neighbourhood_cleansed + host_listings_count + 
                   host_has_profile_pic + host_identity_verified + accommodates + 
                   bedrooms + price + minimum_nights  + cancellation_policy + 
                   `Wireless Internet` + `Air Conditioning` + Kitchen + `Pets live on this property` + 
                   `Family/Kid Friendly` + Washer + `Fire Extinguisher` + Essentials + 
                   Shampoo + Iron + `Safety Card` + Breakfast + `Laptop Friendly Workspace` + 
                   `Elevator in Building` + Doorman, family = propodds, data = listingsReduced)

LLf = logLik(vglmb.reg)
LL0 = logLik(vglm0)

r2 = (1 - (LLf/LL0)) #Adjusted R^2 is 0.0366


#Stepwise selection
ord.step = step(ord.null, scope = list(lower = ord.null, upper = ord.full), direction = "both")

ordstep.reg = orm(formula = review_scores_rating ~ Essentials + neighbourhood_cleansed + 
                    Washer + host_listings_count + Kitchen + price + property_type + 
                    Doorman + host_identity_verified + host_response_time + host_response_rate + 
                    `Family/Kid Friendly` + `Air Conditioning` + `Hair Dryer` + 
                    cancellation_policy + `Fire Extinguisher` + guests_included + 
                    extra_people + bedrooms + accommodates + Shampoo + `Pets Allowed` + 
                    Heating, data = listingsReduced)

#McFadden pseudo R-squared
vglm0 = vglm(review_scores_rating ~ 1, family = propodds, data = listingsReduced)

vglmstep = vglm(formula = review_scores_rating ~ Essentials + neighbourhood_cleansed + 
                  Washer + host_listings_count + Kitchen + price + property_type + 
                  Doorman + host_identity_verified + host_response_time + host_response_rate + 
                  `Family/Kid Friendly` + `Air Conditioning` + `Hair Dryer` + 
                  cancellation_policy + `Fire Extinguisher` + guests_included + 
                  extra_people + bedrooms + accommodates + Shampoo + `Pets Allowed` + 
                  Heating, family = propodds, data = listingsReduced)


LLf3 = logLik(vglmstep)
LL03 = logLik(vglm0)

r2 = (1 - (LLf3/LL03)) #Adjusted R^2 is 0.0403

###logistic regression
####stepwise model selection
s.null <- glm(review_scores_rating~1, data=listingsBucketed, family = 'binomial')
s.full <- glm(review_scores_rating~., data=listingsBucketed, family = 'binomial')


###generalized additive models
#bs = cs: a shrinkage version of cubic regression splines.
#We use cubic splines over thin-plate splines because we don't think there's really a lot of groups in play;
#a shrinkage version is used because we suspect that not all of these variables are important.
#shrinkage also effectively replaces forward/backward/stepwise variable selection
t <- as.character(terms(review_scores_rating ~., data = listingsReducedClean))[3]
t2 <- gsub('\\\n   ', '',t)
t3 <- (strsplit(t2, split = ' \\+ '))[[1]]
t3 <- unlist(lapply(t3, function(x) gsub(' ', '', x)))

nameConverter <- function(x) {
  if (is.factor(listingsReducedClean[,x]) | is.logical(listingsReducedClean[,x]) | x %in% c('bedrooms', 'bathrooms')) {
  return(x)
    } else {
    result <- (paste('s(',x,", bs = 'cs', k = -1)", sep = '')) #auto select knots
  }
  return(result)
}

t4 <- unlist(lapply(t3, nameConverter))
t5 <- paste(t4, collapse = ' + ', sep = '')
t5
#copy above
gam.listings <- gam(review_scores_rating ~ s(host_since, bs = 'cs', k = -1) + host_response_time + s(host_response_rate, bs = 'cs', k = -1) + s(host_acceptance_rate, bs = 'cs', k = -1) + neighbourhood_cleansed + s(host_listings_count, bs = 'cs', k = -1) + host_has_profile_pic + host_identity_verified + property_type + room_type + s(accommodates, bs = 'cs', k = -1) + bathrooms + bedrooms + s(beds, bs = 'cs', k = -1) + bed_type + s(price, bs = 'cs', k = -1) + s(guests_included, bs = 'cs', k = -1) + s(extra_people, bs = 'cs', k = -1) + s(minimum_nights, bs = 'cs', k = -1) + s(maximum_nights, bs = 'cs', k = -1) + cancellation_policy + s(hostLength, bs = 'cs', k = -1) + TV + Internet + WirelessInternet + AirConditioning + Kitchen + PetsAllowed + Petsliveonthisproperty + Dogs + Heating + FamilyKidFriendly + Washer + Dryer + SmokeDetector + CarbonMonoxideDetector + FireExtinguisher + Essentials + Shampoo + LockonBedroomDoor + Hangers + HairDryer + Iron + CableTV + FreeParkingonPremises + FirstAidKit + SafetyCard + Gym + Breakfast + IndoorFireplace + LaptopFriendlyWorkspace + HourCheckin + Cats + HotTub + BuzzerWirelessIntercom + Otherpets + SmokingAllowed + SuitableforEvents + WheelchairAccessible + Doorman + ElevatorinBuilding + Pool + FreeParkingonStreet + PaidParkingOffPremises,
                    data = listingsReducedClean)
plot(gam.listings$fitted.values, gam.listings$residuals)

#cross validation of gam
keptVariables <- t3
keptVariables <- c(t3, 'review_scores_rating')

#get the dataframe with dummy variables
modelingMatrix <- model.matrix(review_scores_rating~., data = listingsReducedClean[,keptVariables])
gamModelingFrame <- as.data.frame(modelingMatrix)
gamModelingFrame$`(Intercept)` <- NULL
gamModelingFrame$review_scores_rating <- listingsReduced$review_scores_rating
names(gamModelingFrame) <- gsub(' |&|-', '', names(gamModelingFrame))

nameConverter2 <- function(x) {
  if (is.factor(gamModelingFrame[,x]) | grepl('TRUE',x) | x %in% c('bedrooms', 'bathrooms') | TRUE %in% lapply(names(Filter(is.factor, listingsReducedClean)), function(y) grepl(y,x))   )  {
    return(x)
  } else {
    result <- (paste('s(',x,", bs = 'cs', k = -1)", sep = '')) #auto select knots
  }
  return(result)
}
#fix factors
factorNames <- names(Filter(is.factor, listingsReducedClean))
factorVec <- lapply(names(gamModelingFrame), function(x) sum(unlist(lapply(factorNames, function(y) grepl(y,x)))) > 0 )
factorVec2 <- lapply(names(gamModelingFrame), function(x) grepl('TRUE',x))
newFactorNames <- names(gamModelingFrame)[unlist(factorVec)]
newFactorNames <- c(newFactorNames, names(gamModelingFrame)[unlist(factorVec2)])
for (i in (1:length(newFactorNames))){
  gamModelingFrame[,newFactorNames[i]] <- as.logical(as.integer(gamModelingFrame[,newFactorNames[i]]))
}

newFormula <- unlist(lapply(names(gamModelingFrame), nameConverter2))
newFormula <- paste(newFormula, collapse = ' + ', sep = '')
newFormula

gam.listings.cv <- CVgam(formula = review_scores_rating ~ s(host_since, bs = 'cs', k = -1) + host_response_timewithinaday + host_response_timewithinafewhours + host_response_timewithinanhour + s(host_response_rate, bs = 'cs', k = -1) + s(host_acceptance_rate, bs = 'cs', k = -1) + neighbourhood_cleansedBackBay + neighbourhood_cleansedBayVillage + neighbourhood_cleansedBeaconHill + neighbourhood_cleansedBrighton + neighbourhood_cleansedCharlestown + neighbourhood_cleansedChinatown + neighbourhood_cleansedDorchester + neighbourhood_cleansedDowntown + neighbourhood_cleansedEastBoston + neighbourhood_cleansedFenway + neighbourhood_cleansedHydePark + neighbourhood_cleansedJamaicaPlain + neighbourhood_cleansedLeatherDistrict + neighbourhood_cleansedLongwoodMedicalArea + neighbourhood_cleansedMattapan + neighbourhood_cleansedMissionHill + neighbourhood_cleansedNorthEnd + neighbourhood_cleansedRoslindale + neighbourhood_cleansedRoxbury + neighbourhood_cleansedSouthBoston + neighbourhood_cleansedSouthBostonWaterfront + neighbourhood_cleansedSouthEnd + neighbourhood_cleansedWestEnd + neighbourhood_cleansedWestRoxbury + s(host_listings_count, bs = 'cs', k = -1) + host_has_profile_pict + host_identity_verifiedt + property_typeApartment + property_typeBedBreakfast + property_typeBoat + property_typeCondominium + property_typeDorm + property_typeEntireFloor + property_typeGuesthouse + property_typeHouse + property_typeLoft + property_typeOther + property_typeTownhouse + property_typeVilla + room_typePrivateroom + room_typeSharedroom + s(accommodates, bs = 'cs', k = -1) + bathrooms + bedrooms + s(beds, bs = 'cs', k = -1) + bed_typeCouch + bed_typeFuton + bed_typePulloutSofa + bed_typeRealBed + s(price, bs = 'cs', k = -1) + s(guests_included, bs = 'cs', k = -1) + s(extra_people, bs = 'cs', k = -1) + s(minimum_nights, bs = 'cs', k = -1) + cancellation_policymoderate + cancellation_policystrict + cancellation_policysuper_strict_30 + s(hostLength, bs = 'cs', k = -1) + TVTRUE + InternetTRUE + WirelessInternetTRUE + AirConditioningTRUE + KitchenTRUE + PetsAllowedTRUE + PetsliveonthispropertyTRUE + DogsTRUE + HeatingTRUE + FamilyKidFriendlyTRUE + WasherTRUE + DryerTRUE + SmokeDetectorTRUE + CarbonMonoxideDetectorTRUE + FireExtinguisherTRUE + EssentialsTRUE + ShampooTRUE + LockonBedroomDoorTRUE + HangersTRUE + HairDryerTRUE + IronTRUE + CableTVTRUE + FreeParkingonPremisesTRUE + FirstAidKitTRUE + SafetyCardTRUE + GymTRUE + BreakfastTRUE + IndoorFireplaceTRUE + LaptopFriendlyWorkspaceTRUE + HourCheckinTRUE + CatsTRUE + HotTubTRUE + BuzzerWirelessIntercomTRUE + OtherpetsTRUE + SmokingAllowedTRUE + SuitableforEventsTRUE + WheelchairAccessibleTRUE + DoormanTRUE + ElevatorinBuildingTRUE + PoolTRUE + FreeParkingonStreetTRUE + PaidParkingOffPremisesTRUE, data = gamModelingFrame, nfold = 10, debug.level = 0, printit = TRUE, method = "GCV.Cp",cvparts = NULL, gamma = 1, seed =100)
                         #data = gamModelingFrame, nfold = 10, debug.level = 0, printit = TRUE, method = "GCV.Cp",cvparts = NULL, gamma = 1, seed =100)
#inclusion of '+ s(maximum_nights, bs = 'cs', k = -1)' causes this model's MSE to explode into the hundred thousands. Removed.
#MSE = 61.2

#R-squared: .26
#AIC: 11321

#3-knot version for comparison purposes
gam.listings2 <- gam(review_scores_rating ~ s(host_since, bs = 'cs', k = 3) + host_response_time + s(host_response_rate, bs = 'cs', k = 3) + s(host_acceptance_rate, bs = 'cs', k = 3) + neighbourhood_cleansed + s(host_listings_count, bs = 'cs', k = 3) + host_has_profile_pic + host_identity_verified + property_type + room_type + s(accommodates, bs = 'cs', k = 3) + bathrooms + bedrooms + s(beds, bs = 'cs', k = 3) + bed_type + s(price, bs = 'cs', k = 3) + s(guests_included, bs = 'cs', k = 3) + s(extra_people, bs = 'cs', k = 3) + s(minimum_nights, bs = 'cs', k = 3) + s(maximum_nights, bs = 'cs', k = 3) + cancellation_policy + s(hostLength, bs = 'cs', k = 3) + TV + Internet + WirelessInternet + AirConditioning + Kitchen + PetsAllowed + Petsliveonthisproperty + Dogs + Heating + FamilyKidFriendly + Washer + Dryer + SmokeDetector + CarbonMonoxideDetector + FireExtinguisher + Essentials + Shampoo + LockonBedroomDoor + Hangers + HairDryer + Iron + CableTV + FreeParkingonPremises + FirstAidKit + SafetyCard + Gym + Breakfast + IndoorFireplace + LaptopFriendlyWorkspace + HourCheckin + Cats + HotTub + BuzzerWirelessIntercom + Otherpets + SmokingAllowed + SuitableforEvents + WheelchairAccessible + Doorman + ElevatorinBuilding + Pool + FreeParkingonStreet + PaidParkingOffPremises,
                    data = listingsReducedClean)
gam.listings2.cv <- CVgam(formula = review_scores_rating ~ s(host_since, bs = 'cs', k = 3) + host_response_timewithinaday + host_response_timewithinafewhours + host_response_timewithinanhour + s(host_response_rate, bs = 'cs', k = 3) + s(host_acceptance_rate, bs = 'cs', k = 3) + neighbourhood_cleansedBackBay + neighbourhood_cleansedBayVillage + neighbourhood_cleansedBeaconHill + neighbourhood_cleansedBrighton + neighbourhood_cleansedCharlestown + neighbourhood_cleansedChinatown + neighbourhood_cleansedDorchester + neighbourhood_cleansedDowntown + neighbourhood_cleansedEastBoston + neighbourhood_cleansedFenway + neighbourhood_cleansedHydePark + neighbourhood_cleansedJamaicaPlain + neighbourhood_cleansedLeatherDistrict + neighbourhood_cleansedLongwoodMedicalArea + neighbourhood_cleansedMattapan + neighbourhood_cleansedMissionHill + neighbourhood_cleansedNorthEnd + neighbourhood_cleansedRoslindale + neighbourhood_cleansedRoxbury + neighbourhood_cleansedSouthBoston + neighbourhood_cleansedSouthBostonWaterfront + neighbourhood_cleansedSouthEnd + neighbourhood_cleansedWestEnd + neighbourhood_cleansedWestRoxbury + s(host_listings_count, bs = 'cs', k = 3) + host_has_profile_pict + host_identity_verifiedt + property_typeApartment + property_typeBedBreakfast + property_typeBoat + property_typeCondominium + property_typeDorm + property_typeEntireFloor + property_typeGuesthouse + property_typeHouse + property_typeLoft + property_typeOther + property_typeTownhouse + property_typeVilla + room_typePrivateroom + room_typeSharedroom + s(accommodates, bs = 'cs', k = 3) + bathrooms + bedrooms + s(beds, bs = 'cs', k = 3) + bed_typeCouch + bed_typeFuton + bed_typePulloutSofa + bed_typeRealBed + s(price, bs = 'cs', k = 3) + s(guests_included, bs = 'cs', k = 3) + s(extra_people, bs = 'cs', k = 3) + s(minimum_nights, bs = 'cs', k = 3) + cancellation_policymoderate + cancellation_policystrict + cancellation_policysuper_strict_30 + s(hostLength, bs = 'cs', k = 3) + TVTRUE + InternetTRUE + WirelessInternetTRUE + AirConditioningTRUE + KitchenTRUE + PetsAllowedTRUE + PetsliveonthispropertyTRUE + DogsTRUE + HeatingTRUE + FamilyKidFriendlyTRUE + WasherTRUE + DryerTRUE + SmokeDetectorTRUE + CarbonMonoxideDetectorTRUE + FireExtinguisherTRUE + EssentialsTRUE + ShampooTRUE + LockonBedroomDoorTRUE + HangersTRUE + HairDryerTRUE + IronTRUE + CableTVTRUE + FreeParkingonPremisesTRUE + FirstAidKitTRUE + SafetyCardTRUE + GymTRUE + BreakfastTRUE + IndoorFireplaceTRUE + LaptopFriendlyWorkspaceTRUE + HourCheckinTRUE + CatsTRUE + HotTubTRUE + BuzzerWirelessIntercomTRUE + OtherpetsTRUE + SmokingAllowedTRUE + SuitableforEventsTRUE + WheelchairAccessibleTRUE + DoormanTRUE + ElevatorinBuildingTRUE + PoolTRUE + FreeParkingonStreetTRUE + PaidParkingOffPremisesTRUE, data = gamModelingFrame, nfold = 10, debug.level = 0, printit = TRUE, method = "GCV.Cp",cvparts = NULL, gamma = 1, seed =100)
#MSE: 65.1845
#R-squared (note that all r-sqaured for gam models are adjusted): .19

#ts version?
gam.listings.cv <- CVgam(formula = review_scores_rating ~ s(host_since, bs = 'ts', k = -1) + host_response_timewithinaday + host_response_timewithinafewhours + host_response_timewithinanhour + s(host_response_rate, bs = 'ts', k = -1) + s(host_acceptance_rate, bs = 'ts', k = -1) + neighbourhood_cleansedBackBay + neighbourhood_cleansedBayVillage + neighbourhood_cleansedBeaconHill + neighbourhood_cleansedBrighton + neighbourhood_cleansedCharlestown + neighbourhood_cleansedChinatown + neighbourhood_cleansedDorchester + neighbourhood_cleansedDowntown + neighbourhood_cleansedEastBoston + neighbourhood_cleansedFenway + neighbourhood_cleansedHydePark + neighbourhood_cleansedJamaicaPlain + neighbourhood_cleansedLeatherDistrict + neighbourhood_cleansedLongwoodMedicalArea + neighbourhood_cleansedMattapan + neighbourhood_cleansedMissionHill + neighbourhood_cleansedNorthEnd + neighbourhood_cleansedRoslindale + neighbourhood_cleansedRoxbury + neighbourhood_cleansedSouthBoston + neighbourhood_cleansedSouthBostonWaterfront + neighbourhood_cleansedSouthEnd + neighbourhood_cleansedWestEnd + neighbourhood_cleansedWestRoxbury + s(host_listings_count, bs = 'ts', k = -1) + host_has_profile_pict + host_identity_verifiedt + property_typeApartment + property_typeBedBreakfast + property_typeBoat + property_typeCondominium + property_typeDorm + property_typeEntireFloor + property_typeGuesthouse + property_typeHouse + property_typeLoft + property_typeOther + property_typeTownhouse + property_typeVilla + room_typePrivateroom + room_typeSharedroom + s(accommodates, bs = 'ts', k = -1) + bathrooms + bedrooms + s(beds, bs = 'ts', k = -1) + bed_typeCouch + bed_typeFuton + bed_typePulloutSofa + bed_typeRealBed + s(price, bs = 'ts', k = -1) + s(guests_included, bs = 'ts', k = -1) + s(extra_people, bs = 'ts', k = -1) + s(minimum_nights, bs = 'ts', k = -1) + cancellation_policymoderate + cancellation_policystrict + cancellation_policysuper_strict_30 + s(hostLength, bs = 'ts', k = -1) + TVTRUE + InternetTRUE + WirelessInternetTRUE + AirConditioningTRUE + KitchenTRUE + PetsAllowedTRUE + PetsliveonthispropertyTRUE + DogsTRUE + HeatingTRUE + FamilyKidFriendlyTRUE + WasherTRUE + DryerTRUE + SmokeDetectorTRUE + CarbonMonoxideDetectorTRUE + FireExtinguisherTRUE + EssentialsTRUE + ShampooTRUE + LockonBedroomDoorTRUE + HangersTRUE + HairDryerTRUE + IronTRUE + CableTVTRUE + FreeParkingonPremisesTRUE + FirstAidKitTRUE + SafetyCardTRUE + GymTRUE + BreakfastTRUE + IndoorFireplaceTRUE + LaptopFriendlyWorkspaceTRUE + HourCheckinTRUE + CatsTRUE + HotTubTRUE + BuzzerWirelessIntercomTRUE + OtherpetsTRUE + SmokingAllowedTRUE + SuitableforEventsTRUE + WheelchairAccessibleTRUE + DoormanTRUE + ElevatorinBuildingTRUE + PoolTRUE + FreeParkingonStreetTRUE + PaidParkingOffPremisesTRUE, data = gamModelingFrame, nfold = 10, debug.level = 0, printit = TRUE, method = "GCV.Cp",cvparts = NULL, gamma = 1, seed =100)
########################Text Mining##############################################################
createCorpus <- function(textCol){
  #cleans a given column and returns a TF-IDF matrix
  hostsums.text = data.frame(matrix(0, nrow(corpusBase), 0))
  
  hostsums.text = cbind(hostsums.text, textCol)
  
  #Rename text column
  names(hostsums.text)[1] = "originalText"
  
  #Read review texts into character vector
  text = as.vector(hostsums.text$originalText)
  
  #Create corpus from text character vector
  hostsums.corpus = Corpus(VectorSource(text))
  
  ###Preprocess corpus
  
  #Convert to lower case
  hostsums.corpus = tm_map(hostsums.corpus, content_transformer(tolower))
  
  #Remove symbols
  toSpace = content_transformer(function(x, pattern) { return (gsub(pattern, " ", x)) })
  
  hostsums.corpus = tm_map(hostsums.corpus, toSpace, '-')
  hostsums.corpus = tm_map(hostsums.corpus, toSpace, '!')
  #hostsums.corpus = tm_map(hostsums.corpus, toSpace, '.')
  hostsums.corpus = tm_map(hostsums.corpus, toSpace, ',')
  hostsums.corpus = tm_map(hostsums.corpus, toSpace, ';')
  hostsums.corpus = tm_map(hostsums.corpus, toSpace, ':')
  hostsums.corpus = tm_map(hostsums.corpus, toSpace, "'")
  
  #writeLines(as.character(reviews.corpus[[1]]))
  
  #Remove punctuation
  hostsums.corpus = tm_map(hostsums.corpus, removePunctuation)
  
  #Remove digits
  hostsums.corpus = tm_map(hostsums.corpus, removeNumbers)
  
  #Remove stopwords and whitespace
  hostsums.corpus = tm_map(hostsums.corpus, removeWords, stopwords("english"))
  hostsums.corpus = tm_map(hostsums.corpus, stripWhitespace)
  
  #Stem the corpus
  hostsums.corpus = tm_map(hostsums.corpus, stemDocument)
  
  return(hostsums.corpus)
}

summariesCorpus <- createCorpus(corpusBase$summary)
profilesCorpus <- createCorpus(corpusBase$host_about)

summariesMatrix <- DocumentTermMatrix(summariesCorpus, control = list(weighting = weightTfIdf))
profilesMatrix <- DocumentTermMatrix(profilesCorpus, control = list(weighting = weightTfIdf))

#################TF-IDF Modeling############################################################
###Summaries field
#prune words used in less than 5% of ads
summariesMatrix <- removeSparseTerms(summariesMatrix, 0.95)

#create model
summaryFrame <- as.data.frame(as.matrix(summariesMatrix))
summaryFrame$review_scores_rating <- listingsReduced$review_scores_rating
s.null <- lm(review_scores_rating~1, data=summaryFrame)
s.full <- lm(review_scores_rating~., data=summaryFrame)
step(s.null, scope=list(lower=s.null, upper=s.full), direction="both")

#cross validate
summary.fit <- lm(formula = review_scores_rating ~ share + build + modern + bus + 
                     min + park + open + welcom + live + public + love + just + 
                     charm + two + condo + walk + space + stop + boston + clean + 
                     studio + new + sunni + bathroom + beauti + end + north + 
                     deck + furnish + featur + perfect, data = summaryFrame)

summarycv <- cv.lm(summaryFrame, summary.fit, m = 10)
#MSE: .766
#R-Squared: .0833

##Host profile field
#prune words used in less than 5% of ads
profilesMatrix <- removeSparseTerms(profilesMatrix, 0.95)

#create model
profileFrame <- as.data.frame(as.matrix(profilesMatrix))
profileFrame$review_scores_rating <- listingsReduced$review_scores_rating
s.null <- lm(review_scores_rating~1, data=profileFrame)
s.full <- lm(review_scores_rating~., data=profileFrame)
step(s.null, scope=list(lower=s.null, upper=s.full), direction="both")

#cross validate
profile.fit <- lm(formula = review_scores_rating ~ meet + vacat + new + see + 
                     term + room + welcom + make + close + fit + place + offer + 
                     full + restaur + spend + time + take + know + one + apart + 
                     now + person + share + enjoy + also + hope + life + trip + 
                     knowledg + provid + servic + old + forward + unit + come + 
                     day, data = profileFrame)
profilecv <- cv.lm(profileFrame, profile.fit, m = 10)
#MSE: .893
#Adjusted R-Squared: 0.138

###generalized additive models
#summary frame
summariesMatrix <- removeSparseTerms(summariesMatrix, 0.90)
summaryFrame <- as.data.frame(as.matrix(summariesMatrix))
summaryFrame$review_scores_rating <- listingsReduced$review_scores_rating
t <- as.character(terms(review_scores_rating ~., data = summaryFrame))[3]
t2 <- gsub('\\\n   ', '',t)
t3 <- (strsplit(t2, split = ' \\+ '))[[1]]
t3 <- unlist(lapply(t3, function(x) gsub(' ', '', x)))

nameConverter <- function(x) {
  return (paste('s(',x,", bs = 'cs', k = 3)", sep = ''))
}

t4 <- unlist(lapply(t3, nameConverter))
t5 <- paste(t4, collapse = ' + ', sep = '')
t5
#copy above
gam.summaries <- gam(review_scores_rating ~ s(access, bs = 'cs', k = 3) + s(apart, bs = 'cs', k = 3) + s(away, bs = 'cs', k = 3) + s(back, bs = 'cs', k = 3) + s(bath, bs = 'cs', k = 3) + s(bathroom, bs = 'cs', k = 3) + s(bay, bs = 'cs', k = 3) + s(beauti, bs = 'cs', k = 3) + s(bed, bs = 'cs', k = 3) + s(bedroom, bs = 'cs', k = 3) + s(boston, bs = 'cs', k = 3) + s(center, bs = 'cs', k = 3) + s(citi, bs = 'cs', k = 3) + s(close, bs = 'cs', k = 3) + s(comfort, bs = 'cs', k = 3) + s(downtown, bs = 'cs', k = 3) + s(easi, bs = 'cs', k = 3) + s(end, bs = 'cs', k = 3) + s(floor, bs = 'cs', k = 3) + s(full, bs = 'cs', k = 3) + s(great, bs = 'cs', k = 3) + s(heart, bs = 'cs', k = 3) + s(histor, bs = 'cs', k = 3) + s(home, bs = 'cs', k = 3) + s(hous, bs = 'cs', k = 3) + s(just, bs = 'cs', k = 3) + s(kitchen, bs = 'cs', k = 3) + s(line, bs = 'cs', k = 3) + s(live, bs = 'cs', k = 3) + s(locat, bs = 'cs', k = 3) + s(min, bs = 'cs', k = 3) + s(minut, bs = 'cs', k = 3) + s(neighborhood, bs = 'cs', k = 3) + s(offer, bs = 'cs', k = 3) + s(one, bs = 'cs', k = 3) + s(park, bs = 'cs', k = 3) + s(place, bs = 'cs', k = 3) + s(privat, bs = 'cs', k = 3) + s(public, bs = 'cs', k = 3) + s(queen, bs = 'cs', k = 3) + s(quiet, bs = 'cs', k = 3) + s(renov, bs = 'cs', k = 3) + s(restaur, bs = 'cs', k = 3) + s(room, bs = 'cs', k = 3) + s(share, bs = 'cs', k = 3) + s(shop, bs = 'cs', k = 3) + s(size, bs = 'cs', k = 3) + s(south, bs = 'cs', k = 3) + s(spacious, bs = 'cs', k = 3) + s(station, bs = 'cs', k = 3) + s(step, bs = 'cs', k = 3) + s(street, bs = 'cs', k = 3) + s(subway, bs = 'cs', k = 3) + s(two, bs = 'cs', k = 3) + s(walk, bs = 'cs', k = 3),
                    data = summaryFrame)
plot(gam.summaries$fitted.values, gam.summaries$residuals)
#cross validate
CVgam(formula=review_scores_rating ~ s(access, bs = 'cs', k = 3) + s(apart, bs = 'cs', k = 3) + s(away, bs = 'cs', k = 3) + s(back, bs = 'cs', k = 3) + s(bath, bs = 'cs', k = 3) + s(bathroom, bs = 'cs', k = 3) + s(bay, bs = 'cs', k = 3) + s(beauti, bs = 'cs', k = 3) + s(bed, bs = 'cs', k = 3) + s(bedroom, bs = 'cs', k = 3) + s(boston, bs = 'cs', k = 3) + s(center, bs = 'cs', k = 3) + s(citi, bs = 'cs', k = 3) + s(close, bs = 'cs', k = 3) + s(comfort, bs = 'cs', k = 3) + s(downtown, bs = 'cs', k = 3) + s(easi, bs = 'cs', k = 3) + s(end, bs = 'cs', k = 3) + s(floor, bs = 'cs', k = 3) + s(full, bs = 'cs', k = 3) + s(great, bs = 'cs', k = 3) + s(heart, bs = 'cs', k = 3) + s(histor, bs = 'cs', k = 3) + s(home, bs = 'cs', k = 3) + s(hous, bs = 'cs', k = 3) + s(just, bs = 'cs', k = 3) + s(kitchen, bs = 'cs', k = 3) + s(line, bs = 'cs', k = 3) + s(live, bs = 'cs', k = 3) + s(locat, bs = 'cs', k = 3) + s(min, bs = 'cs', k = 3) + s(minut, bs = 'cs', k = 3) + s(neighborhood, bs = 'cs', k = 3) + s(offer, bs = 'cs', k = 3) + s(one, bs = 'cs', k = 3) + s(park, bs = 'cs', k = 3) + s(place, bs = 'cs', k = 3) + s(privat, bs = 'cs', k = 3) + s(public, bs = 'cs', k = 3) + s(queen, bs = 'cs', k = 3) + s(quiet, bs = 'cs', k = 3) + s(renov, bs = 'cs', k = 3) + s(restaur, bs = 'cs', k = 3) + s(room, bs = 'cs', k = 3) + s(share, bs = 'cs', k = 3) + s(shop, bs = 'cs', k = 3) + s(size, bs = 'cs', k = 3) + s(south, bs = 'cs', k = 3) + s(spacious, bs = 'cs', k = 3) + s(station, bs = 'cs', k = 3) + s(step, bs = 'cs', k = 3) + s(street, bs = 'cs', k = 3) + s(subway, bs = 'cs', k = 3) + s(two, bs = 'cs', k = 3) + s(walk, bs = 'cs', k = 3), data = summaryFrame, nfold = 10, debug.level = 0, printit = TRUE, method = "GCV.Cp",cvparts = NULL, gamma = 1, seed =100)
#MSE: 70.0001
#R-squared: .0675
#11592

#profile frame
profilesMatrix <- removeSparseTerms(profilesMatrix, 0.90)
profilesFrame <- as.data.frame(as.matrix(profilesMatrix))
profilesFrame$review_scores_rating <- listingsReduced$review_scores_rating
t <- as.character(terms(review_scores_rating ~., data = profilesFrame))[3]
t2 <- gsub('\\\n   ', '',t)
t3 <- (strsplit(t2, split = ' \\+ '))[[1]]
t3 <- unlist(lapply(t3, function(x) gsub(' ', '', x)))

nameConverter <- function(x) {
  return (paste('s(',x,", bs = 'cs', k = 3)", sep = ''))
}

t4 <- unlist(lapply(t3, nameConverter))
t5 <- paste(t4, collapse = ' + ', sep = '')
t5
#copy above
gam.profiles <- gam(review_scores_rating ~ s(airbnb, bs = 'cs', k = 3) + s(also, bs = 'cs', k = 3) + s(apart, bs = 'cs', k = 3) + s(area, bs = 'cs', k = 3) + s(around, bs = 'cs', k = 3) + s(best, bs = 'cs', k = 3) + s(boston, bs = 'cs', k = 3) + s(busi, bs = 'cs', k = 3) + s(can, bs = 'cs', k = 3) + s(citi, bs = 'cs', k = 3) + s(comfort, bs = 'cs', k = 3) + s(cook, bs = 'cs', k = 3) + s(enjoy, bs = 'cs', k = 3) + s(experi, bs = 'cs', k = 3) + s(famili, bs = 'cs', k = 3) + s(food, bs = 'cs', k = 3) + s(free, bs = 'cs', k = 3) + s(friend, bs = 'cs', k = 3) + s(get, bs = 'cs', k = 3) + s(good, bs = 'cs', k = 3) + s(great, bs = 'cs', k = 3) + s(guest, bs = 'cs', k = 3) + s(happi, bs = 'cs', k = 3) + s(help, bs = 'cs', k = 3) + s(home, bs = 'cs', k = 3) + s(host, bs = 'cs', k = 3) + s(hous, bs = 'cs', k = 3) + s(interest, bs = 'cs', k = 3) + s(just, bs = 'cs', k = 3) + s(know, bs = 'cs', k = 3) + s(like, bs = 'cs', k = 3) + s(live, bs = 'cs', k = 3) + s(local, bs = 'cs', k = 3) + s(locat, bs = 'cs', k = 3) + s(look, bs = 'cs', k = 3) + s(love, bs = 'cs', k = 3) + s(make, bs = 'cs', k = 3) + s(manag, bs = 'cs', k = 3) + s(mani, bs = 'cs', k = 3) + s(meet, bs = 'cs', k = 3) + s(need, bs = 'cs', k = 3) + s(new, bs = 'cs', k = 3) + s(offer, bs = 'cs', k = 3) + s(one, bs = 'cs', k = 3) + s(peopl, bs = 'cs', k = 3) + s(place, bs = 'cs', k = 3) + s(profession, bs = 'cs', k = 3) + s(provid, bs = 'cs', k = 3) + s(rental, bs = 'cs', k = 3) + s(servic, bs = 'cs', k = 3) + s(short, bs = 'cs', k = 3) + s(stay, bs = 'cs', k = 3) + s(time, bs = 'cs', k = 3) + s(travel, bs = 'cs', k = 3) + s(welcom, bs = 'cs', k = 3) + s(well, bs = 'cs', k = 3) + s(will, bs = 'cs', k = 3) + s(work, bs = 'cs', k = 3) + s(world, bs = 'cs', k = 3) + s(year, bs = 'cs', k = 3),
                     data = profilesFrame)
plot(gam.profiles$fitted.values, gam.profiles$residuals)
#cross validate
CVgam(formula=review_scores_rating ~ s(airbnb, bs = 'cs', k = 3) + s(also, bs = 'cs', k = 3) + s(apart, bs = 'cs', k = 3) + s(area, bs = 'cs', k = 3) + s(around, bs = 'cs', k = 3) + s(best, bs = 'cs', k = 3) + s(boston, bs = 'cs', k = 3) + s(busi, bs = 'cs', k = 3) + s(can, bs = 'cs', k = 3) + s(citi, bs = 'cs', k = 3) + s(comfort, bs = 'cs', k = 3) + s(cook, bs = 'cs', k = 3) + s(enjoy, bs = 'cs', k = 3) + s(experi, bs = 'cs', k = 3) + s(famili, bs = 'cs', k = 3) + s(food, bs = 'cs', k = 3) + s(free, bs = 'cs', k = 3) + s(friend, bs = 'cs', k = 3) + s(get, bs = 'cs', k = 3) + s(good, bs = 'cs', k = 3) + s(great, bs = 'cs', k = 3) + s(guest, bs = 'cs', k = 3) + s(happi, bs = 'cs', k = 3) + s(help, bs = 'cs', k = 3) + s(home, bs = 'cs', k = 3) + s(host, bs = 'cs', k = 3) + s(hous, bs = 'cs', k = 3) + s(interest, bs = 'cs', k = 3) + s(just, bs = 'cs', k = 3) + s(know, bs = 'cs', k = 3) + s(like, bs = 'cs', k = 3) + s(live, bs = 'cs', k = 3) + s(local, bs = 'cs', k = 3) + s(locat, bs = 'cs', k = 3) + s(look, bs = 'cs', k = 3) + s(love, bs = 'cs', k = 3) + s(make, bs = 'cs', k = 3) + s(manag, bs = 'cs', k = 3) + s(mani, bs = 'cs', k = 3) + s(meet, bs = 'cs', k = 3) + s(need, bs = 'cs', k = 3) + s(new, bs = 'cs', k = 3) + s(offer, bs = 'cs', k = 3) + s(one, bs = 'cs', k = 3) + s(peopl, bs = 'cs', k = 3) + s(place, bs = 'cs', k = 3) + s(profession, bs = 'cs', k = 3) + s(provid, bs = 'cs', k = 3) + s(rental, bs = 'cs', k = 3) + s(servic, bs = 'cs', k = 3) + s(short, bs = 'cs', k = 3) + s(stay, bs = 'cs', k = 3) + s(time, bs = 'cs', k = 3) + s(travel, bs = 'cs', k = 3) + s(welcom, bs = 'cs', k = 3) + s(well, bs = 'cs', k = 3) + s(will, bs = 'cs', k = 3) + s(work, bs = 'cs', k = 3) + s(world, bs = 'cs', k = 3) + s(year, bs = 'cs', k = 3), data = profilesFrame, nfold = 10, debug.level = 0, printit = TRUE, method = "GCV.Cp",cvparts = NULL, gamma = 1, seed =100)
#MSE: 65.1208
#R-squared: .161
#AIC: 11426

#################LDA Modeling###############################################################
#first check to see if coherent categories are produced at all
#if yes, attempt modeling

summaries.dtm <- DocumentTermMatrix(summariesCorpus)
summaries.lda <- LDA(summaries.dtm, 5)
summaries.terms <- as.matrix(terms(summaries.lda,20))
#Trying a few different number of topics doesn't really yield terms that seem semantically distinct to (admittedly casual) inspection
#no sense modeling any further- even if they're predictive, they don't give actionable feedback to prospective sellers.

profiles.dtm <- DocumentTermMatrix(profilesCorpus)
profiles.lda <- LDA(profiles.dtm, 2)
profiles.terms <- as.matrix(terms(profiles.lda,20))
#Two seems to be about the sweet spot for this one- topic 2 is very business-oriented, with words that relate what the host can do for 
#the client like 'offer','furnish','provide','comfort','need', etc.
#By contrast, topic 1 is much more about the host themself, with words like 'live','travel','work','love','host'

#modeling profile topics
profiles.topics <- as.data.frame(as.matrix(topics(profiles.lda)))
profiles.topics$review_scores_rating <- listingsReduced$review_scores_rating
colnames(profiles.topics)[1] <- 'profileTopic'
profiles.topics$profileTopic <- as.factor(profiles.topics$profileTopic)
profilesModel <- lm(review_scores_rating~., data = profiles.topics)
cv.lm(profiles.topics, profilesModel, m = 10)
#MSE: 69.4
#R-squared: .0213

#modeling profile topic probabilities
profiles.probabilities <- as.data.frame(profiles.lda@gamma)#as.data.frame(as.matrix(topics(profiles.lda)))
profiles.probabilities$review_scores_rating <- listingsReduced$review_scores_rating
names(profiles.probabilities) <- c('LDA_profile_personal','LDA_profile_impersonal','review_scores_rating')
profileProbabilitiessModel <- lm(review_scores_rating~., data = profiles.probabilities)
cv.lm(profiles.probabilities, profileProbabilitiessModel, m = 10)
#MSE: .983
#Adjusted R-squared: .0194

#spline model of topic probabilities
gam.topic <- gam(review_scores_rating ~ s(LDA_profile_personal, bs = 'cs', k = -1) + s(LDA_profile_impersonal, bs = 'cs', k = -1), data = profiles.probabilities)
gam.topic.cv <- CVgam(formula = review_scores_rating ~ s(personal, bs = 'cs', k = -1) + s(impersonal, bs = 'cs', k = -1), data = profiles.probabilities, nfold = 10, debug.level = 0, printit = TRUE, method = "GCV.Cp",cvparts = NULL, gamma = 1, seed =100)
#MSE: 68.9
#R-squared: .0285
#AIC: 11639
#################Integrated Modeling / Final Analysis########################################################
#Throw it all in a model, variable select, cross validate, and go to bed. Look at it again when I'm not a zombie.
#Or make Greg deal with it.

#create combined frame (without topic)
profileVars <- profileFrame[,c('meet','vacat', 'new', 'see','term', 'room', 'welcom', 'make', 'close', 'fit', 'place',
                 'offer','full', 'restaur', 'spend', 'time', 'take', 'know', 'one', 'apart', 'now', 'person',
                 'share', 'enjoy', 'also', 'hope', 'life', 'trip','knowledg', 'provid', 'servic','old', 
                 'forward','unit', 'come','day')]
names(profileVars) <- paste('profile',names(profileVars))
summaryVars <- summaryFrame[,c('share','build','modern','bus','min','park','open','welcom','live','public',
                               'love','just','charm','two','condo','walk','space','stop','boston','clean',
                               'studio','new','sunni','bathroom','beauti','end','north','deck','furnish','featur','perfect')]
names(summaryVars) <- paste('summary',names(summaryVars))
combinedFrame <- cbind(listingsReduced, profileVars)
combinedFrame <- cbind(combinedFrame, summaryVars)

#perform stepwise selection
s.null <- lm(review_scores_rating~1, data=combinedFrame)
s.full <- lm(review_scores_rating~., data=combinedFrame)
stepwiseSelection <- step(s.null, scope=list(lower=s.null, upper=s.full), direction="both")

#get the indexing vector
keptVariables <- as.character(stepwiseSelection$call)[2]
keptVariables <- gsub('~','+',keptVariables)
keptVariables <- gsub('`','',keptVariables)
keptVariables <- gsub('\\\\','',keptVariables)
keptVariables <- strsplit(keptVariables,'\\+')
keptVariables <- as.vector(keptVariables)[[1]]
keptVariables <- trimws(keptVariables, 'both')
names(combinedFrame) <- gsub('`','',names(combinedFrame))

#fit and cross validate
stepwise.fit <- lm(review_scores_rating ~., data = combinedFrame[,keptVariables])
stepwisecv <- cv.lm(combinedFrame, stepwise.fit, m = 10)
#Adjusted R-squared: 0.2338
#MSE: .811


#the LDA part is a bit methodologically shaky (we're assuming we understand the difference between the topics), so let's validate it separately
combinedFrameWithTopics <- cbind(combinedFrame, profiles.topics$profileTopic)

#perform stepwise selection
s.null <- lm(review_scores_rating~1, data=combinedFrameWithTopics)
s.full <- lm(review_scores_rating~., data=combinedFrameWithTopics)
stepwiseSelection <- step(s.null, scope=list(lower=s.null, upper=s.full), direction="both")

#get the indexing vector
keptVariables <- as.character(stepwiseSelection$call)[2]
keptVariables <- gsub('~','+',keptVariables)
keptVariables <- gsub('`','',keptVariables)
keptVariables <- gsub('\\\\','',keptVariables)
keptVariables <- strsplit(keptVariables,'\\+')
keptVariables <- as.vector(keptVariables)[[1]]
keptVariables <- trimws(keptVariables, 'both')

#fit and cross validate
stepwise.fit <- lm(review_scores_rating ~., data = combinedFrameWithTopics[,keptVariables])
stepwisecv <- cv.lm(combinedFrameWithTopics, stepwise.fit, m = 10)
#adjusted R-squared: .239
#MSE: .811

#combined GAM model
profileVars <- profilesFrame[, -which(names(profilesFrame) == "review_scores_rating")]
names(profileVars) <- paste('profile',names(profileVars), sep = '_')
summaryVars <- summaryFrame[, -which(names(summaryFrame) == "review_scores_rating")]
names(summaryVars) <- paste('summary',names(summaryVars), sep = '_')
LDAVars <- profiles.probabilities[, -which(names(profiles.probabilities) == "review_scores_rating")]
finalFrame <- cbind(gamModelingFrame, profileVars)
finalFrame <- cbind(finalFrame, LDAVars)
finalFrame <- cbind(finalFrame, summaryVars)
#remember to remove that one var
finalFrame$maximum_nights <- NULL

#formula, add/remove review scores

nameConverter3 <- function(x) {
  if (is.logical(finalFrame[,x])   )  {
    return(x)
  } else {
    result <- (paste('s(',x,", bs = 'cs', k = -1)", sep = '')) #auto select knots
  }
  return(result)
}

newFormula <- unlist(lapply(names(finalFrame), nameConverter3))
newFormula <- paste(newFormula, collapse = ' + ', sep = '')
newFormula

#errors when using profile_love
gam.combined <- gam(review_scores_rating ~ s(host_since, bs = 'cs', k = 3) + host_response_timewithinaday + host_response_timewithinafewhours + host_response_timewithinanhour + s(host_response_rate, bs = 'cs', k = 3) + s(host_acceptance_rate, bs = 'cs', k = 3) + neighbourhood_cleansedBackBay + neighbourhood_cleansedBayVillage + neighbourhood_cleansedBeaconHill + neighbourhood_cleansedBrighton + neighbourhood_cleansedCharlestown + neighbourhood_cleansedChinatown + neighbourhood_cleansedDorchester + neighbourhood_cleansedDowntown + neighbourhood_cleansedEastBoston + neighbourhood_cleansedFenway + neighbourhood_cleansedHydePark + neighbourhood_cleansedJamaicaPlain + neighbourhood_cleansedLeatherDistrict + neighbourhood_cleansedLongwoodMedicalArea + neighbourhood_cleansedMattapan + neighbourhood_cleansedMissionHill + neighbourhood_cleansedNorthEnd + neighbourhood_cleansedRoslindale + neighbourhood_cleansedRoxbury + neighbourhood_cleansedSouthBoston + neighbourhood_cleansedSouthBostonWaterfront + neighbourhood_cleansedSouthEnd + neighbourhood_cleansedWestEnd + neighbourhood_cleansedWestRoxbury + s(host_listings_count, bs = 'cs', k = 3) + host_has_profile_pict + host_identity_verifiedt + property_typeApartment + property_typeBedBreakfast + property_typeBoat + property_typeCondominium + property_typeDorm + property_typeEntireFloor + property_typeGuesthouse + property_typeHouse + property_typeLoft + property_typeOther + property_typeTownhouse + property_typeVilla + room_typePrivateroom + room_typeSharedroom + s(accommodates, bs = 'cs', k = 3) + s(bathrooms, bs = 'cs', k = 3) + s(bedrooms, bs = 'cs', k = 3) + s(beds, bs = 'cs', k = 3) + bed_typeCouch + bed_typeFuton + bed_typePulloutSofa + bed_typeRealBed + s(price, bs = 'cs', k = 3) + s(guests_included, bs = 'cs', k = 3) + s(extra_people, bs = 'cs', k = 3) + s(minimum_nights, bs = 'cs', k = 3) + cancellation_policymoderate + cancellation_policystrict + cancellation_policysuper_strict_30 + s(hostLength, bs = 'cs', k = 3) + TVTRUE + InternetTRUE + WirelessInternetTRUE + AirConditioningTRUE + KitchenTRUE + PetsAllowedTRUE + PetsliveonthispropertyTRUE + DogsTRUE + HeatingTRUE + FamilyKidFriendlyTRUE + WasherTRUE + DryerTRUE + SmokeDetectorTRUE + CarbonMonoxideDetectorTRUE + FireExtinguisherTRUE + EssentialsTRUE + ShampooTRUE + LockonBedroomDoorTRUE + HangersTRUE + HairDryerTRUE + IronTRUE + CableTVTRUE + FreeParkingonPremisesTRUE + FirstAidKitTRUE + SafetyCardTRUE + GymTRUE + BreakfastTRUE + IndoorFireplaceTRUE + LaptopFriendlyWorkspaceTRUE + HourCheckinTRUE + CatsTRUE + HotTubTRUE + BuzzerWirelessIntercomTRUE + OtherpetsTRUE + SmokingAllowedTRUE + SuitableforEventsTRUE + WheelchairAccessibleTRUE + DoormanTRUE + ElevatorinBuildingTRUE + PoolTRUE + FreeParkingonStreetTRUE + PaidParkingOffPremisesTRUE + s(profile_airbnb, bs = 'cs', k = 3) + s(profile_also, bs = 'cs', k = 3) + s(profile_apart, bs = 'cs', k = 3) + s(profile_area, bs = 'cs', k = 3) + s(profile_around, bs = 'cs', k = 3) + s(profile_best, bs = 'cs', k = 3) + s(profile_boston, bs = 'cs', k = 3) + s(profile_busi, bs = 'cs', k = 3) + s(profile_can, bs = 'cs', k = 3) + s(profile_citi, bs = 'cs', k = 3) + s(profile_comfort, bs = 'cs', k = 3) + s(profile_cook, bs = 'cs', k = 3) + s(profile_enjoy, bs = 'cs', k = 3) + s(profile_experi, bs = 'cs', k = 3) + s(profile_famili, bs = 'cs', k = 3) + s(profile_food, bs = 'cs', k = 3) + s(profile_free, bs = 'cs', k = 3) + s(profile_friend, bs = 'cs', k = 3) + s(profile_get, bs = 'cs', k = 3) + s(profile_good, bs = 'cs', k = 3) + s(profile_great, bs = 'cs', k = 3) + s(profile_guest, bs = 'cs', k = 3) + s(profile_happi, bs = 'cs', k = 3) + s(profile_help, bs = 'cs', k = 3) + s(profile_home, bs = 'cs', k = 3) + s(profile_host, bs = 'cs', k = 3) + s(profile_hous, bs = 'cs', k = 3) + s(profile_interest, bs = 'cs', k = 3) + s(profile_just, bs = 'cs', k = 3) + s(profile_know, bs = 'cs', k = 3) + s(profile_like, bs = 'cs', k = 3) + s(profile_live, bs = 'cs', k = 3) + s(profile_local, bs = 'cs', k = 3) + s(profile_locat, bs = 'cs', k = 3) + s(profile_look, bs = 'cs', k = 3) + #s(profile_love, bs = 'cs', k = 3) +
                      s(profile_make, bs = 'cs', k = 3) + s(profile_manag, bs = 'cs', k = 3) + s(profile_mani, bs = 'cs', k = 3) + s(profile_meet, bs = 'cs', k = 3) +
                      s(profile_need, bs = 'cs', k = 3) + s(profile_new, bs = 'cs', k = 3) + s(profile_offer, bs = 'cs', k = 3) + s(profile_one, bs = 'cs', k = 3) + s(profile_peopl, bs = 'cs', k = 3) + s(profile_place, bs = 'cs', k = 3) + s(profile_profession, bs = 'cs', k = 3) + s(profile_provid, bs = 'cs', k = 3) + s(profile_rental, bs = 'cs', k = 3) + s(profile_servic, bs = 'cs', k = 3) + s(profile_short, bs = 'cs', k = 3) + s(profile_stay, bs = 'cs', k = 3) + s(profile_time, bs = 'cs', k = 3) + s(profile_travel, bs = 'cs', k = 3) + s(profile_welcom, bs = 'cs', k = 3) + s(profile_well, bs = 'cs', k = 3) + s(profile_will, bs = 'cs', k = 3) + s(profile_work, bs = 'cs', k = 3) + s(profile_world, bs = 'cs', k = 3) + s(profile_year, bs = 'cs', k = 3) + s(LDA_profile_personal, bs = 'cs', k = 3) + s(LDA_profile_impersonal, bs = 'cs', k = 3) + s(summary_access, bs = 'cs', k = 3) + s(summary_apart, bs = 'cs', k = 3) + s(summary_away, bs = 'cs', k = 3) + s(summary_back, bs = 'cs', k = 3) + s(summary_bath, bs = 'cs', k = 3) + s(summary_bathroom, bs = 'cs', k = 3) + s(summary_bay, bs = 'cs', k = 3) + s(summary_beauti, bs = 'cs', k = 3) + s(summary_bed, bs = 'cs', k = 3) + s(summary_bedroom, bs = 'cs', k = 3) + s(summary_boston, bs = 'cs', k = 3) + s(summary_center, bs = 'cs', k = 3) + s(summary_citi, bs = 'cs', k = 3) + s(summary_close, bs = 'cs', k = 3) + s(summary_comfort, bs = 'cs', k = 3) + s(summary_downtown, bs = 'cs', k = 3) + s(summary_easi, bs = 'cs', k = 3) + s(summary_end, bs = 'cs', k = 3) + s(summary_floor, bs = 'cs', k = 3) + s(summary_full, bs = 'cs', k = 3) + s(summary_great, bs = 'cs', k = 3) + s(summary_heart, bs = 'cs', k = 3) + s(summary_histor, bs = 'cs', k = 3) + s(summary_home, bs = 'cs', k = 3) + s(summary_hous, bs = 'cs', k = 3) + s(summary_just, bs = 'cs', k = 3) + s(summary_kitchen, bs = 'cs', k = 3) + s(summary_line, bs = 'cs', k = 3) + s(summary_live, bs = 'cs', k = 3) + s(summary_locat, bs = 'cs', k = 3) + s(summary_min, bs = 'cs', k = 3) + s(summary_minut, bs = 'cs', k = 3) + s(summary_neighborhood, bs = 'cs', k = 3) + s(summary_offer, bs = 'cs', k = 3) + s(summary_one, bs = 'cs', k = 3) + s(summary_park, bs = 'cs', k = 3) + s(summary_place, bs = 'cs', k = 3) + s(summary_privat, bs = 'cs', k = 3) + s(summary_public, bs = 'cs', k = 3) + s(summary_queen, bs = 'cs', k = 3) + s(summary_quiet, bs = 'cs', k = 3) + s(summary_renov, bs = 'cs', k = 3) + s(summary_restaur, bs = 'cs', k = 3) + s(summary_room, bs = 'cs', k = 3) + s(summary_share, bs = 'cs', k = 3) + s(summary_shop, bs = 'cs', k = 3) + s(summary_size, bs = 'cs', k = 3) + s(summary_south, bs = 'cs', k = 3) + s(summary_spacious, bs = 'cs', k = 3) + s(summary_station, bs = 'cs', k = 3) + s(summary_step, bs = 'cs', k = 3) + s(summary_street, bs = 'cs', k = 3) + s(summary_subway, bs = 'cs', k = 3) + s(summary_two, bs = 'cs', k = 3) + s(summary_walk, bs = 'cs', k = 3),
                    data = finalFrame)
#Adjusted R-squared: 0.287 *3 knots
#MSE: 66.5182

#CVgam
cv.combined <- CVgam(formula = review_scores_rating ~ s(host_since, bs = 'cs', k = 3) + host_response_timewithinaday + host_response_timewithinafewhours + host_response_timewithinanhour + s(host_response_rate, bs = 'cs', k = 3) + s(host_acceptance_rate, bs = 'cs', k = 3) + neighbourhood_cleansedBackBay + neighbourhood_cleansedBayVillage + neighbourhood_cleansedBeaconHill + neighbourhood_cleansedBrighton + neighbourhood_cleansedCharlestown + neighbourhood_cleansedChinatown + neighbourhood_cleansedDorchester + neighbourhood_cleansedDowntown + neighbourhood_cleansedEastBoston + neighbourhood_cleansedFenway + neighbourhood_cleansedHydePark + neighbourhood_cleansedJamaicaPlain + neighbourhood_cleansedLeatherDistrict + neighbourhood_cleansedLongwoodMedicalArea + neighbourhood_cleansedMattapan + neighbourhood_cleansedMissionHill + neighbourhood_cleansedNorthEnd + neighbourhood_cleansedRoslindale + neighbourhood_cleansedRoxbury + neighbourhood_cleansedSouthBoston + neighbourhood_cleansedSouthBostonWaterfront + neighbourhood_cleansedSouthEnd + neighbourhood_cleansedWestEnd + neighbourhood_cleansedWestRoxbury + s(host_listings_count, bs = 'cs', k = 3) + host_has_profile_pict + host_identity_verifiedt + property_typeApartment + property_typeBedBreakfast + property_typeBoat + property_typeCondominium + property_typeDorm + property_typeEntireFloor + property_typeGuesthouse + property_typeHouse + property_typeLoft + property_typeOther + property_typeTownhouse + property_typeVilla + room_typePrivateroom + room_typeSharedroom + s(accommodates, bs = 'cs', k = 3) + s(bathrooms, bs = 'cs', k = 3) + s(bedrooms, bs = 'cs', k = 3) + s(beds, bs = 'cs', k = 3) + bed_typeCouch + bed_typeFuton + bed_typePulloutSofa + bed_typeRealBed + s(price, bs = 'cs', k = 3) + s(guests_included, bs = 'cs', k = 3) + s(extra_people, bs = 'cs', k = 3) + s(minimum_nights, bs = 'cs', k = 3) + cancellation_policymoderate + cancellation_policystrict + cancellation_policysuper_strict_30 + s(hostLength, bs = 'cs', k = 3) + TVTRUE + InternetTRUE + WirelessInternetTRUE + AirConditioningTRUE + KitchenTRUE + PetsAllowedTRUE + PetsliveonthispropertyTRUE + DogsTRUE + HeatingTRUE + FamilyKidFriendlyTRUE + WasherTRUE + DryerTRUE + SmokeDetectorTRUE + CarbonMonoxideDetectorTRUE + FireExtinguisherTRUE + EssentialsTRUE + ShampooTRUE + LockonBedroomDoorTRUE + HangersTRUE + HairDryerTRUE + IronTRUE + CableTVTRUE + FreeParkingonPremisesTRUE + FirstAidKitTRUE + SafetyCardTRUE + GymTRUE + BreakfastTRUE + IndoorFireplaceTRUE + LaptopFriendlyWorkspaceTRUE + HourCheckinTRUE + CatsTRUE + HotTubTRUE + BuzzerWirelessIntercomTRUE + OtherpetsTRUE + SmokingAllowedTRUE + SuitableforEventsTRUE + WheelchairAccessibleTRUE + DoormanTRUE + ElevatorinBuildingTRUE + PoolTRUE + FreeParkingonStreetTRUE + PaidParkingOffPremisesTRUE + s(profile_airbnb, bs = 'cs', k = 3) + s(profile_also, bs = 'cs', k = 3) + s(profile_apart, bs = 'cs', k = 3) + s(profile_area, bs = 'cs', k = 3) + s(profile_around, bs = 'cs', k = 3) + s(profile_best, bs = 'cs', k = 3) + s(profile_boston, bs = 'cs', k = 3) + s(profile_busi, bs = 'cs', k = 3) + s(profile_can, bs = 'cs', k = 3) + s(profile_citi, bs = 'cs', k = 3) + s(profile_comfort, bs = 'cs', k = 3) + s(profile_cook, bs = 'cs', k = 3) + s(profile_enjoy, bs = 'cs', k = 3) + s(profile_experi, bs = 'cs', k = 3) + s(profile_famili, bs = 'cs', k = 3) + s(profile_food, bs = 'cs', k = 3) + s(profile_free, bs = 'cs', k = 3) + s(profile_friend, bs = 'cs', k = 3) + s(profile_get, bs = 'cs', k = 3) + s(profile_good, bs = 'cs', k = 3) + s(profile_great, bs = 'cs', k = 3) + s(profile_guest, bs = 'cs', k = 3) + s(profile_happi, bs = 'cs', k = 3) + s(profile_help, bs = 'cs', k = 3) + s(profile_home, bs = 'cs', k = 3) + s(profile_host, bs = 'cs', k = 3) + s(profile_hous, bs = 'cs', k = 3) + s(profile_interest, bs = 'cs', k = 3) + s(profile_just, bs = 'cs', k = 3) + s(profile_know, bs = 'cs', k = 3) + s(profile_like, bs = 'cs', k = 3) + s(profile_live, bs = 'cs', k = 3) + s(profile_local, bs = 'cs', k = 3) + s(profile_locat, bs = 'cs', k = 3) + s(profile_look, bs = 'cs', k = 3) + #s(profile_love, bs = 'cs', k = 3) +
        s(profile_make, bs = 'cs', k = 3) + s(profile_manag, bs = 'cs', k = 3) + s(profile_mani, bs = 'cs', k = 3) + s(profile_meet, bs = 'cs', k = 3) +
        s(profile_need, bs = 'cs', k = 3) + s(profile_new, bs = 'cs', k = 3) + s(profile_offer, bs = 'cs', k = 3) + s(profile_one, bs = 'cs', k = 3) + s(profile_peopl, bs = 'cs', k = 3) + s(profile_place, bs = 'cs', k = 3) + s(profile_profession, bs = 'cs', k = 3) + s(profile_provid, bs = 'cs', k = 3) + s(profile_rental, bs = 'cs', k = 3) + s(profile_servic, bs = 'cs', k = 3) + s(profile_short, bs = 'cs', k = 3) + s(profile_stay, bs = 'cs', k = 3) + s(profile_time, bs = 'cs', k = 3) + s(profile_travel, bs = 'cs', k = 3) + s(profile_welcom, bs = 'cs', k = 3) + s(profile_well, bs = 'cs', k = 3) + s(profile_will, bs = 'cs', k = 3) + s(profile_work, bs = 'cs', k = 3) + s(profile_world, bs = 'cs', k = 3) + s(profile_year, bs = 'cs', k = 3) + s(LDA_profile_personal, bs = 'cs', k = 3) + s(LDA_profile_impersonal, bs = 'cs', k = 3) + s(summary_access, bs = 'cs', k = 3) + s(summary_apart, bs = 'cs', k = 3) + s(summary_away, bs = 'cs', k = 3) + s(summary_back, bs = 'cs', k = 3) + s(summary_bath, bs = 'cs', k = 3) + s(summary_bathroom, bs = 'cs', k = 3) + s(summary_bay, bs = 'cs', k = 3) + s(summary_beauti, bs = 'cs', k = 3) + s(summary_bed, bs = 'cs', k = 3) + s(summary_bedroom, bs = 'cs', k = 3) + s(summary_boston, bs = 'cs', k = 3) + s(summary_center, bs = 'cs', k = 3) + s(summary_citi, bs = 'cs', k = 3) + s(summary_close, bs = 'cs', k = 3) + s(summary_comfort, bs = 'cs', k = 3) + s(summary_downtown, bs = 'cs', k = 3) + s(summary_easi, bs = 'cs', k = 3) + s(summary_end, bs = 'cs', k = 3) + s(summary_floor, bs = 'cs', k = 3) + s(summary_full, bs = 'cs', k = 3) + s(summary_great, bs = 'cs', k = 3) + s(summary_heart, bs = 'cs', k = 3) + s(summary_histor, bs = 'cs', k = 3) + s(summary_home, bs = 'cs', k = 3) + s(summary_hous, bs = 'cs', k = 3) + s(summary_just, bs = 'cs', k = 3) + s(summary_kitchen, bs = 'cs', k = 3) + s(summary_line, bs = 'cs', k = 3) + s(summary_live, bs = 'cs', k = 3) + s(summary_locat, bs = 'cs', k = 3) + s(summary_min, bs = 'cs', k = 3) + s(summary_minut, bs = 'cs', k = 3) + s(summary_neighborhood, bs = 'cs', k = 3) + s(summary_offer, bs = 'cs', k = 3) + s(summary_one, bs = 'cs', k = 3) + s(summary_park, bs = 'cs', k = 3) + s(summary_place, bs = 'cs', k = 3) + s(summary_privat, bs = 'cs', k = 3) + s(summary_public, bs = 'cs', k = 3) + s(summary_queen, bs = 'cs', k = 3) + s(summary_quiet, bs = 'cs', k = 3) + s(summary_renov, bs = 'cs', k = 3) + s(summary_restaur, bs = 'cs', k = 3) + s(summary_room, bs = 'cs', k = 3) + s(summary_share, bs = 'cs', k = 3) + s(summary_shop, bs = 'cs', k = 3) + s(summary_size, bs = 'cs', k = 3) + s(summary_south, bs = 'cs', k = 3) + s(summary_spacious, bs = 'cs', k = 3) + s(summary_station, bs = 'cs', k = 3) + s(summary_step, bs = 'cs', k = 3) + s(summary_street, bs = 'cs', k = 3) + s(summary_subway, bs = 'cs', k = 3) + s(summary_two, bs = 'cs', k = 3) + s(summary_walk, bs = 'cs', k = 3), data = finalFrame, nfold = 10, debug.level = 0, printit = TRUE, method = "GCV.Cp",cvparts = NULL, gamma = 1, seed =100)

#cv knot selection
gam.combined <- gam(review_scores_rating ~ s(host_since, bs = 'cs', k = -1) + host_response_timewithinaday + host_response_timewithinafewhours + host_response_timewithinanhour + s(host_response_rate, bs = 'cs', k = -1) + s(host_acceptance_rate, bs = 'cs', k = -1) + neighbourhood_cleansedBackBay + neighbourhood_cleansedBayVillage + neighbourhood_cleansedBeaconHill + neighbourhood_cleansedBrighton + neighbourhood_cleansedCharlestown + neighbourhood_cleansedChinatown + neighbourhood_cleansedDorchester + neighbourhood_cleansedDowntown + neighbourhood_cleansedEastBoston + neighbourhood_cleansedFenway + neighbourhood_cleansedHydePark + neighbourhood_cleansedJamaicaPlain + neighbourhood_cleansedLeatherDistrict + neighbourhood_cleansedLongwoodMedicalArea + neighbourhood_cleansedMattapan + neighbourhood_cleansedMissionHill + neighbourhood_cleansedNorthEnd + neighbourhood_cleansedRoslindale + neighbourhood_cleansedRoxbury + neighbourhood_cleansedSouthBoston + neighbourhood_cleansedSouthBostonWaterfront + neighbourhood_cleansedSouthEnd + neighbourhood_cleansedWestEnd + neighbourhood_cleansedWestRoxbury + s(host_listings_count, bs = 'cs', k = -1) + host_has_profile_pict + host_identity_verifiedt + property_typeApartment + property_typeBedBreakfast + property_typeBoat + property_typeCondominium + property_typeDorm + property_typeEntireFloor + property_typeGuesthouse + property_typeHouse + property_typeLoft + property_typeOther + property_typeTownhouse + property_typeVilla + room_typePrivateroom + room_typeSharedroom + s(accommodates, bs = 'cs', k = -1) + s(bathrooms, bs = 'cs', k = -1) + s(bedrooms, bs = 'cs', k = -1) + s(beds, bs = 'cs', k = -1) + bed_typeCouch + bed_typeFuton + bed_typePulloutSofa + bed_typeRealBed + s(price, bs = 'cs', k = -1) + s(guests_included, bs = 'cs', k = -1) + s(extra_people, bs = 'cs', k = -1) + s(minimum_nights, bs = 'cs', k = -1) + cancellation_policymoderate + cancellation_policystrict + cancellation_policysuper_strict_30 + s(hostLength, bs = 'cs', k = -1) + TVTRUE + InternetTRUE + WirelessInternetTRUE + AirConditioningTRUE + KitchenTRUE + PetsAllowedTRUE + PetsliveonthispropertyTRUE + DogsTRUE + HeatingTRUE + FamilyKidFriendlyTRUE + WasherTRUE + DryerTRUE + SmokeDetectorTRUE + CarbonMonoxideDetectorTRUE + FireExtinguisherTRUE + EssentialsTRUE + ShampooTRUE + LockonBedroomDoorTRUE + HangersTRUE + HairDryerTRUE + IronTRUE + CableTVTRUE + FreeParkingonPremisesTRUE + FirstAidKitTRUE + SafetyCardTRUE + GymTRUE + BreakfastTRUE + IndoorFireplaceTRUE + LaptopFriendlyWorkspaceTRUE + HourCheckinTRUE + CatsTRUE + HotTubTRUE + BuzzerWirelessIntercomTRUE + OtherpetsTRUE + SmokingAllowedTRUE + SuitableforEventsTRUE + WheelchairAccessibleTRUE + DoormanTRUE + ElevatorinBuildingTRUE + PoolTRUE + FreeParkingonStreetTRUE + PaidParkingOffPremisesTRUE + s(profile_airbnb, bs = 'cs', k = -1) + s(profile_also, bs = 'cs', k = -1) + s(profile_apart, bs = 'cs', k = -1) + s(profile_area, bs = 'cs', k = -1) + s(profile_around, bs = 'cs', k = -1) + s(profile_best, bs = 'cs', k = -1) + s(profile_boston, bs = 'cs', k = -1) + s(profile_busi, bs = 'cs', k = -1) + s(profile_can, bs = 'cs', k = -1) + s(profile_citi, bs = 'cs', k = -1) + s(profile_comfort, bs = 'cs', k = -1) + s(profile_cook, bs = 'cs', k = -1) + s(profile_enjoy, bs = 'cs', k = -1) + s(profile_experi, bs = 'cs', k = -1) + s(profile_famili, bs = 'cs', k = -1) + s(profile_food, bs = 'cs', k = -1) + s(profile_free, bs = 'cs', k = -1) + s(profile_friend, bs = 'cs', k = -1) + s(profile_get, bs = 'cs', k = -1) + s(profile_good, bs = 'cs', k = -1) + s(profile_great, bs = 'cs', k = -1) + s(profile_guest, bs = 'cs', k = -1) + s(profile_happi, bs = 'cs', k = -1) + s(profile_help, bs = 'cs', k = -1) + s(profile_home, bs = 'cs', k = -1) + s(profile_host, bs = 'cs', k = -1) + s(profile_hous, bs = 'cs', k = -1) + s(profile_interest, bs = 'cs', k = -1) + s(profile_just, bs = 'cs', k = -1) + s(profile_know, bs = 'cs', k = -1) + s(profile_like, bs = 'cs', k = -1) + s(profile_live, bs = 'cs', k = -1) + s(profile_local, bs = 'cs', k = -1) + s(profile_locat, bs = 'cs', k = -1) + s(profile_look, bs = 'cs', k = -1) + #s(profile_love, bs = 'cs', k = -1) +
                      s(profile_make, bs = 'cs', k = -1) + s(profile_manag, bs = 'cs', k = -1) + s(profile_mani, bs = 'cs', k = -1) + s(profile_meet, bs = 'cs', k = -1) +
                      s(profile_need, bs = 'cs', k = -1) + s(profile_new, bs = 'cs', k = -1) + s(profile_offer, bs = 'cs', k = -1) + s(profile_one, bs = 'cs', k = -1) + s(profile_peopl, bs = 'cs', k = -1) + s(profile_place, bs = 'cs', k = -1) + s(profile_profession, bs = 'cs', k = -1) + s(profile_provid, bs = 'cs', k = -1) + s(profile_rental, bs = 'cs', k = -1) + s(profile_servic, bs = 'cs', k = -1) + s(profile_short, bs = 'cs', k = -1) + s(profile_stay, bs = 'cs', k = -1) + s(profile_time, bs = 'cs', k = -1) + s(profile_travel, bs = 'cs', k = -1) + s(profile_welcom, bs = 'cs', k = -1) + s(profile_well, bs = 'cs', k = -1) + s(profile_will, bs = 'cs', k = -1) + s(profile_work, bs = 'cs', k = -1) + s(profile_world, bs = 'cs', k = -1) + s(profile_year, bs = 'cs', k = -1) + s(LDA_profile_personal, bs = 'cs', k = -1) + s(LDA_profile_impersonal, bs = 'cs', k = -1) + s(summary_access, bs = 'cs', k = -1) + s(summary_apart, bs = 'cs', k = -1) + s(summary_away, bs = 'cs', k = -1) + s(summary_back, bs = 'cs', k = -1) + s(summary_bath, bs = 'cs', k = -1) + s(summary_bathroom, bs = 'cs', k = -1) + s(summary_bay, bs = 'cs', k = -1) + s(summary_beauti, bs = 'cs', k = -1) + s(summary_bed, bs = 'cs', k = -1) + s(summary_bedroom, bs = 'cs', k = -1) + s(summary_boston, bs = 'cs', k = -1) + s(summary_center, bs = 'cs', k = -1) + s(summary_citi, bs = 'cs', k = -1) + s(summary_close, bs = 'cs', k = -1) + s(summary_comfort, bs = 'cs', k = -1) + s(summary_downtown, bs = 'cs', k = -1) + s(summary_easi, bs = 'cs', k = -1) + s(summary_end, bs = 'cs', k = -1) + s(summary_floor, bs = 'cs', k = -1) + s(summary_full, bs = 'cs', k = -1) + s(summary_great, bs = 'cs', k = -1) + s(summary_heart, bs = 'cs', k = -1) + s(summary_histor, bs = 'cs', k = -1) + s(summary_home, bs = 'cs', k = -1) + s(summary_hous, bs = 'cs', k = -1) + s(summary_just, bs = 'cs', k = -1) + s(summary_kitchen, bs = 'cs', k = -1) + s(summary_line, bs = 'cs', k = -1) + s(summary_live, bs = 'cs', k = -1) + s(summary_locat, bs = 'cs', k = -1) + s(summary_min, bs = 'cs', k = -1) + s(summary_minut, bs = 'cs', k = -1) + s(summary_neighborhood, bs = 'cs', k = -1) + s(summary_offer, bs = 'cs', k = -1) + s(summary_one, bs = 'cs', k = -1) + s(summary_park, bs = 'cs', k = -1) + s(summary_place, bs = 'cs', k = -1) + s(summary_privat, bs = 'cs', k = -1) + s(summary_public, bs = 'cs', k = -1) + s(summary_queen, bs = 'cs', k = -1) + s(summary_quiet, bs = 'cs', k = -1) + s(summary_renov, bs = 'cs', k = -1) + s(summary_restaur, bs = 'cs', k = -1) + s(summary_room, bs = 'cs', k = -1) + s(summary_share, bs = 'cs', k = -1) + s(summary_shop, bs = 'cs', k = -1) + s(summary_size, bs = 'cs', k = -1) + s(summary_south, bs = 'cs', k = -1) + s(summary_spacious, bs = 'cs', k = -1) + s(summary_station, bs = 'cs', k = -1) + s(summary_step, bs = 'cs', k = -1) + s(summary_street, bs = 'cs', k = -1) + s(summary_subway, bs = 'cs', k = -1) + s(summary_two, bs = 'cs', k = -1) + s(summary_walk, bs = 'cs', k = -1),
                    data = finalFrame)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              