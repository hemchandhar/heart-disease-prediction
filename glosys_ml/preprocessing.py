import pandas as pd

class Heart_preprocessing:
	def preprocessing(h):
		df=pd.read_csv(h,header=None)
				#DATASET DESCRIPTION
				#SAMPLES-303 , FEATURES -15
				#INPUT LABELS(14 FEATURES)
						# age - age in years
						# sex - sex (1 = male; 0 = female)
						# cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)
						# trestbps - resting blood pressure (in mm Hg on admission to the hospital)
						# chol - serum cholestoral in mg/dl
						# fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
						# restecg - resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)
						# thalach - maximum heart rate achieved
						# exang - exercise induced angina (1 = yes; 0 = no)
						# oldpeak - ST depression induced by exercise relative to rest
						# slope - the slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)
						# ca - number of major vessels (0-3) colored by flourosopy
						# thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
						# num - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < 50% diameter narrowing; Value 1 = > 50% diameter narrowing)
				#OUPUT LABELS(1 FEATURES)
						#target(0-no , 1-yes)

		df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
	              'fbs', 'restecg', 'thalach', 'exang', 
	              'oldpeak', 'slope', 'ca', 'thal', 'target']


		### 1 = male, 0 = female
		df.isnull().sum()

		df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
		#df['sex'] = df.sex.map({0: 'female', 1: 'male'})
		df['thal'] = df.thal.fillna(df.thal.mean())
		df['ca'] = df.ca.fillna(df.ca.mean())
		return df