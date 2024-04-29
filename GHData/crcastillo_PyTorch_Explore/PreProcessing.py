"""
    Data PreProcessing

    Objective: Create a standard set of transformers to use as part of the scikit-learn Pipeline class.

    Notes:
        -
"""

# <editor-fold desc="Load libraries and dataset, set parameters">

# Load required modules
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


# <editor-fold desc="Create custom transformers for Pipeline">

# Create custom transformer that converts to str
class ObjectConversion_Manual(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, Object_List: list):
        self.Object_List = Object_List
    # Return self
    def fit(self, X, y=None):
        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y=None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Iterate through Object_List to convert fields to str then np.object
        for i in self.Object_List:
            Store = np.where(
                pd.isnull(X_Copy[i])
                , X_Copy[i]
                , X_Copy[i].astype(str)
                ).copy()
            X_Copy[i] = Store
        # Return the copied DataFrame with the fixed field
        return X_Copy

# Create custom transformer that drops features with a percent missing exceeding a defined cutoff
class FeatureMissingness(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, cutoff: float):
        self.cutoff = cutoff
    # Return self
    def fit(self, X, y = None):
        # Create a DataFrame with the percent missing for each field
        self.X_Missing = pd.DataFrame(X.isnull().sum().sort_values(ascending=False) / X.shape[0])
        # Identify the fields to drop
        self.X_Missing_Fields = self.X_Missing[self.X_Missing[0] > self.cutoff].index.to_list()

        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Return the copied DataFrame
        return X.loc[:, ~X.columns.isin(self.X_Missing_Fields)]


# Create custom transformer that coerces categorical levels to np.nan per a user defined dictionary key-value pairing
class NA_Manual(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, NA_Manual_Dict: dict):
        self.NA_Manual_Dict = NA_Manual_Dict
    # Return self
    def fit(self, X, y = None):
        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Iterate through NA_Manual_Dic to coerce specified categorical levels to np.nan
        for i in self.NA_Manual_Dict:
            X_Copy.loc[X_Copy[i].isin(self.NA_Manual_Dict[i]), i] = np.nan
        # Return the copied DataFrame with the fixed field
        return X_Copy

# Create custom transformer that fills NaN fields per a user defined dictionary key-value pairing
class Fill_NA_Manual(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, Fill_NA_Dict: dict):
        self.Fill_NA_Dict = Fill_NA_Dict
    # Return self
    def fit(self, X, y = None):
        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Iterate through Fill_NA_Dict to fillna with desired value
        for i in self.Fill_NA_Dict:
            Store = X_Copy[i].fillna(self.Fill_NA_Dict[i]).copy()
            X_Copy[i] = Store
        # Return the copied DataFrame with the fixed field
        return X_Copy

# Create custom transformer that drops defined features
class HardExclude(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, Exclude_Fields: list):
        self.Exclude_Fields = Exclude_Fields
    # Return self
    def fit(self, X, y = None):
        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create copy of DF while excluding the defined fields
        X_Copy = X.loc[:, ~X.columns.isin(self.Exclude_Fields)].copy()
        # Return the copied DataFrame
        return X_Copy

# Create custom transformer that collapses low observation categories into _Other_ based on a user provided dictionary
class LowObsCounts_Manual(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, Fill_Other_Dict: dict):
        self.Fill_Other_Dict = Fill_Other_Dict
    # Return self
    def fit(self, X, y = None):
        # Instantiate dictionary to map category transforms
        self.Transform_Dict = {}
        # Iterate through Fill_Other_Dict to coerce categories below specified threshold with _Other_
        for i in self.Fill_Other_Dict:
            # Store Series object with proportions of non-na levels
            Low_Obs_Column = X[i].value_counts().sort_values(ascending=True) / X[i].count()
            # Store low observation categories
            Low_Obs_Categories = Low_Obs_Column.index[Low_Obs_Column < self.Fill_Other_Dict[i]].to_list()
            # Populate the Transform_Dict with the categories for each variable
            self.Transform_Dict[i] = Low_Obs_Categories

        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Iterate through Fill_Other_Dict to coerce categories below specified threshold with _Other_
        for Iter in self.Transform_Dict:
            # Replace low observation categories with _Other_
            X_Copy.loc[X_Copy[Iter].isin(self.Transform_Dict[Iter]), Iter] = '_Other_'
        # Return the copied DataFrame with the fixed field
        return X_Copy

# Create custom transformer that transforms defined variables with log(x + 1)
class LogPlusOne_Manual(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, Transform_Fields: list):
        self.Transform_Fields = Transform_Fields
    # Return self
    def fit(self, X, y = None):
        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Define transformation function
        def LogPlusOne_Transform(x):
            if np.isnan(x):
                return np.nan
            elif x > 0:
                return np.log(x + 1)
            else:
                return np.log(1)
        # Iterate through Fill_Other_Dict to coerce categories below specified threshold with _Other_
        for i in self.Transform_Fields:
            # Store a copy of the field and apply the transform
            Transform_Field = X_Copy[i].apply(LogPlusOne_Transform).copy()
            # Overwrite the transformed field
            X_Copy[i] = Transform_Field
        # Return the copied DataFrame
        return X_Copy

# Create custom transformer that identifies correlation pairs above threshold and removes one with highest mean abs corr
class FindCorrelation(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, threshold = 0.95):
        self.threshold = threshold
    # Return self
    def fit(self, X, y = None):
        # Store matrix of correlation coefficients
        self.corr_mat = np.corrcoef(X.T)
        # Store absolute value correlation matrix
        abs_corr_mat = np.abs(self.corr_mat)
        # Store the lower diagonal
        self.corr_mat = np.tril(m = self.corr_mat, k = -1)
        # Instantiate the array to store columns to remove
        self.Remove = []
        # Iterate through columns of input matrix to check for pairs above threshold
        for Col in range(X.shape[1]):
            # Store the row element that exceeds threshold
            Row = np.where(np.abs(self.corr_mat[:, Col]) > self.threshold)[0]
            if len(Row) > 0:
                # Append Row to Remove with lowest mean corr
                if np.mean(abs_corr_mat[:, Col]) > np.mean(abs_corr_mat[Row[0]]):
                    self.Remove.append(Row[0])
                else:
                    self.Remove.append(Col)
        # Eliminate duplicates
        self.Remove = np.unique(self.Remove)
        # Store vector of False booleans
        Remove_Bool = np.zeros(shape = X.shape[1], dtype = bool)
        # Check to see if Remove is empty
        if len(self.Remove) != 0:
            # Use self.Remove to switch corresponding elements to True
            Remove_Bool[self.Remove] = True
        # Store final vector of columns to remove
        self.Remove_Vector = Remove_Bool

        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create a copy of the matrix without highly correlated columns
        return X.T[~self.Remove_Vector].T

# Create custom transformer that converts all object fields to category fields
class CategoryConverter(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self):
        self
    # Return self
    def fit(self, X, y = None):
        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Iterate through Fill_Other_Dict to coerce categories below specified threshold with _Other_
        for i in X_Copy.select_dtypes('O').columns:
            # Store a copy of the field and apply the transform to category
            Transform_Field = X_Copy[i].astype('category').cat.remove_unused_categories().copy()
            # Overwrite the transformed field
            X_Copy[i] = Transform_Field
        # Return the copied DataFrame
        return X_Copy

# Create custom transformer that coerces all new categorical levels to NaN
class NewCategoryLevels(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self):
        self
    # Return self
    def fit(self, X, y = None):
        # Instantiate a dictionary to store categorical levels
        self.category_levels = {}
        # Identify category fields and store their respective levels
        for i in X.select_dtypes('category').columns:
            self.category_levels[i] = X[i].cat._parent._dtype._categories
        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Iterate through self.category_levels to set levels accordingly and coerce new levels to NaN
        for i in self.category_levels.keys():
            # Store a copy of the field and set the category levels according to fit
            Transform_Field = X_Copy[i].cat.set_categories(self.category_levels[i])
            # Overwrite the transformed field
            X_Copy[i] = Transform_Field
        # Return the copied DataFrame
        return X_Copy

# Create custom transformer that identifies correlation pairs above threshold and removes one with highest mean abs corr
class VIFScreen(BaseEstimator, TransformerMixin):
    # Class constructor
    def __init__(self, threshold = 10.0):
        self.threshold = threshold
    # Return self
    def fit(self, X, y = None):
        # Retrieve numeric data from X and assign a Constant field
        Numeric_Data = pd.DataFrame(
            data = X
            , dtype='float64'
        ).assign(Constant = 1)
        # Store VIF data
        self.VIF = pd.Series(
            data = [variance_inflation_factor(Numeric_Data.values, i)
                   for i in range(Numeric_Data.shape[1])]
            , index = Numeric_Data.columns
        ).sort_values(ascending = False)
        # Drop the constant field
        self.VIF.drop(labels = 'Constant', inplace = True)
        # Store the columns to remove
        self.Remove = self.VIF[self.VIF > self.threshold].index.to_list()
        # Store vector of False booleans
        Remove_Bool = np.zeros(shape = X.shape[1], dtype = bool)
        # Check to see if Remove is empty
        if len(self.Remove) != 0:
            # Use self.Remove to switch corresponding elements to True
            Remove_Bool[self.Remove] = True
        # Store final vector of columns to remove
        self.Remove_Vector = Remove_Bool

        return self
    # Method that describes what we need the transformer to do
    def transform(self, X, y = None):
        # Create a copy of the matrix without highly correlated columns
        return X.T[~self.Remove_Vector].T

# Create custom transformer identifies columns that has missing values and creates new column with a missing value
# indicator
class IndicateMissing(TransformerMixin, BaseEstimator):
    def __init__(self, missing_values=np.nan):
        self.missing_values = missing_values

    def fit(self, X, y = None):
        # Instantiate list to store columns that contain NaN
        self.NaN_Columns = []
        # Iterate through all columns looking for ones that contain a NaN (or NA)
        for col in X.columns:
            if X[col].isna().sum() > 0:
                self.NaN_Columns.append(col)
        return self

    def transform(self, X, y = None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Iterate through self.category_levels to set levels accordingly and coerce new levels to NaN
        for col in self.NaN_Columns:
            # Store a copy of the field and set the category levels according to fit
            NaN_Check = X_Copy[col].isna().astype('category')
            # Overwrite the transformed field
            X_Copy[str(col + '_NaN')] = NaN_Check
        # Return the copied DataFrame
        return X_Copy

# Create custom transformer that can perform label encoding and store the classes in a callable dictionary
class ModifiedLabelEncoder(TransformerMixin, BaseEstimator):
    def __init__(self):
        self

    def fit(self, X, y = None):
        # Instantiate dictionary to store label encoders that we'll fit
        self.labels_dict = {}
        # Iterate through all columns looking for ones that contain a NaN (or NA)
        for col in range(X.shape[1]):
            self.labels_dict[col] = LabelEncoder().fit(X[:,col])
        return self

    def transform(self, X, y = None):
        # Create a copy of the dataframe
        X_Copy = X.copy()
        # Iterate through self.category_levels to set levels accordingly and coerce new levels to NaN
        for key in self.labels_dict.keys():
            # Store a copy of the field and set the category levels according to fit
            Transform_Field = self.labels_dict[key].transform(X_Copy[:,key])
            # Overwrite the transformed field
            X_Copy[:,key] = Transform_Field
        # Return the copied DataFrame
        return X_Copy

# </editor-fold>

# <editor-fold desc="Define final Numeric/Categorical transformers for Pipeline">

# Define Numeric transformations
Numeric_Transformer = Pipeline(
    steps = [
        ('SimpleImputer with median', SimpleImputer(
            strategy = 'median'
            , verbose = 1))
        , ('Remove highly intra-correlated fields' , FindCorrelation(threshold = 0.95))
        # , ('Run VIF screen' ,VIFScreen(threshold = 10.0))
        , ('StandardScaler', StandardScaler())
    ]
    , verbose = True
)

# Define Numeric transformations
Categorical_Transformer = Pipeline(
    steps = [
        ('SimpleImputer with most_frequent', SimpleImputer(
            strategy = 'most_frequent'
            , verbose = 1))
        , ('OneHotEncoding', OneHotEncoder(
            handle_unknown = 'error'
            , drop = 'first'
            , sparse = False))
    ]
    , verbose = True
)

# Combine the Numeric and Categorical transformers
Combined_Transformer = ColumnTransformer(
    transformers = [
        ('Numeric transformations'
         , Numeric_Transformer
         , make_column_selector(dtype_include = np.number))
        , ('Categorical transformations'
           , Categorical_Transformer
           , make_column_selector(dtype_include = 'category'))
    ]
    , remainder = 'passthrough'
)

# </editor-fold>

# <editor-fold desc="Create Processing Pipeline and apply to Train/Test">

# Craft a Pre-Processing Model Pipeline
Pipeline_PreProcessing = Pipeline(
    steps = [
        # ('Convert specified fields to str', ObjectConversion_Manual(
        #     Object_List = [
        #         'bDeath'
        #     ]))
        # ,
        ('Remove high missing fields', FeatureMissingness(cutoff = 0.95))
        , ('Transform defined numerical fields with log(x + 1)' , LogPlusOne_Manual(
            Transform_Fields = [
                'duration'
            ]))
        , ('Set defined categorical levels to NaN', NA_Manual(
            NA_Manual_Dict = {
                'pdays': [
                    -1
                ]
            })
           )
        # , ('Fill the NA records for defined fields', Fill_NA_Manual(
        #     Fill_NA_Dict = {
        #         'Annuity_Count' : 0
        #     })
        #    )
        # , ('Remove defined fields', HardExclude(
        #     Exclude_Fields = [
        #         'ID'
        #         , 'chBIO_UserType'  # Low variance since this just equals Client
        #     ])
        #    )
        # , ('Coerce categorical fields with low observation counts to _Other_' , LowObsCounts_Manual(
        #     Fill_Other_Dict = {
        #         'chBIO_Occupation': 0.005
        #     })
        #    )
        , ('Convert all object fields to category fields', CategoryConverter())
        , ('Coerce novel categorical levels to NaN', NewCategoryLevels())
        , ('Numeric/Categorical transforms', Combined_Transformer)
        , ('Variance screen' , VarianceThreshold(threshold = 0))
    ]
    , verbose = True
)

# </editor-fold>

# <editor-fold desc="Build function that instantiates a PreProcessing Pipeline">

def CreatePreProcessingPipeline(verbose = False):
    # Define Numeric transformations
    Numeric_Transformer = Pipeline(
        steps = [
            ('SimpleImputer with median', SimpleImputer(
                strategy = 'median'
                , verbose = 1))
            , ('Remove highly intra-correlated fields' , FindCorrelation(threshold = 0.95))
            # , ('Run VIF screen' ,VIFScreen(threshold = 10.0))
            , ('StandardScaler', StandardScaler())
        ]
        , verbose = verbose
    )
    # Define Numeric transformations
    Categorical_Transformer = Pipeline(
        steps = [
            ('SimpleImputer with most_frequent', SimpleImputer(
                strategy = 'most_frequent'
                , verbose = 1))
            , ('LabelEncoder', ModifiedLabelEncoder())
            , ('OneHotEncoding', OneHotEncoder(
                handle_unknown = 'error'
                , drop = 'first'
                , sparse = False))
        ]
        , verbose = verbose
    )
    # Combine the Numeric and Categorical transformers
    Combined_Transformer = ColumnTransformer(
        transformers = [
            ('Numeric transformations'
             , Numeric_Transformer
             , make_column_selector(dtype_include = np.number))
            , ('Categorical transformations'
               , Categorical_Transformer
               , make_column_selector(dtype_include = 'category'))
        ]
        , remainder = 'passthrough'
    )

    # Craft a Pre-Processing Model Pipeline
    Pipeline_PreProcessing = Pipeline(
        steps = [
            # ('Convert specified fields to str', ObjectConversion_Manual(
            #     Object_List = [
            #         'bDeath'
            #     ]))
            # ,
            ('Remove high missing fields', FeatureMissingness(cutoff = 0.95))
            , ('Transform defined numerical fields with log(x + 1)' , LogPlusOne_Manual(
                Transform_Fields = [
                    'duration'
                ]))
            , ('Set defined categorical levels to NaN', NA_Manual(
                NA_Manual_Dict = {
                    'pdays': [
                        -1
                    ]
                })
               )
            # , ('Fill the NA records for defined fields', Fill_NA_Manual(
            #     Fill_NA_Dict = {
            #         'Annuity_Count' : 0
            #     })
            #    )
            # , ('Remove defined fields', HardExclude(
            #     Exclude_Fields = [
            #         'ID'
            #         , 'chBIO_UserType'  # Low variance since this just equals Client
            #     ])
            #    )
            # , ('Coerce categorical fields with low observation counts to _Other_' , LowObsCounts_Manual(
            #     Fill_Other_Dict = {
            #         'chBIO_Occupation': 0.005
            #     })
            #    )
            , ('Convert all object fields to category fields', CategoryConverter())
            , ('Coerce novel categorical levels to NaN', NewCategoryLevels())
            , ('MissingValueIndicator' , IndicateMissing())
            , ('Numeric/Categorical transforms', Combined_Transformer)
            , ('Variance screen' , VarianceThreshold(threshold = 0))
        ]
        , verbose = verbose
    )

    return Pipeline_PreProcessing

# </editor-fold>