import datetime
import pickle
import string

import pandas as pd
import numpy as np
from bson.binary import Binary
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

from api.exceptions import NoModel
from utils.io import model_deserialize, model_serialize
from utils.metrics import cosine_sim
from utils.misc import get_traceback, logger, sort_tuple

from ..basic_model import basic_model


class lsa_v1_0_0(basic_model):
    def __init__(self):
        self.model_name = 'lsa'
        self.model_version = 'v1.0.0'

    def __get_max_k__(self):
        return 95

    def __max_past_orders__(self):
        return 15

    def __get_data__(self, db_main):
        print('MongoDB connection established!')
        pipeline1 = [
                    {'$match': {'date': {'$gt':'2016-01-01'}}},
                    {'$project': {
                                    'date': 1,
                                    'recipe_id':1,
                                    'user_id':1,
                                    'rating':1
                                }
                    },
                    {'$sort': {'date':-1}}
                ]
        users, items, time, ratings = [], [], [], []
        for item in db_main.orders_data.aggregate(pipeline1):
            users.append(item.get('user_id'))
            items.append(item.get('recipe_id'))
            time.append(item.get('date'))
            ratings.append(item.get('rating'))
        orders = pd.DataFrame({'user_id':users, 'recipe_id':items, 'time':time, 'rating':ratings})
        items_list = orders['recipe_id'].unique()

        pipeline2 = [
                        {'$project': {
                                        'name':1,
                                        'id':1,
                                        'steps':1,
                                        'ingredients':1
                                    }
                        }
                    ]
        names, ids, recipe, ings = [], [], [], []
        for item in db_main.item_collection.aggregate(pipeline2):
            names.append(item.get('name'))
            ids.append(item.get('id'))
            recipe.append(item.get('steps'))
            ings.append(item.get('ingredients'))
        item_data = pd.DataFrame({'item':names, 'recipe_id':ids, 'recipe':recipe, 'ingredients':ings})

        item_data = item_data[item_data['recipe_id'].isin(items_list)]

        data = pd.merge(orders, item_data, how='left', on=['recipe_id'])
        print('Data imported!')
        return data

    def __get_food_content__(self, raw_data):

        raw_data['steps_'] = raw_data['recipe'].apply(lambda x: ' '.join(eval(x)).replace(',', ''))
        raw_data['steps_'] = raw_data['steps_'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        raw_data['ingredients_'] = raw_data['ingredients'].apply(lambda x: ' '.join(eval(x)))
        raw_data['details'] = raw_data['steps_']+' '+raw_data['ingredients_']
        raw_data.drop(['recipe', 'ingredients'], axis=1, inplace=True)

        food_list = raw_data['details'].unique().tolist()
        food_ids_list = raw_data['recipe_id'].unique().tolist()

        return (food_ids_list, food_list)

    def __pre_processor__(self, details):
        print('processing data...')
        ignore_chars = ''',:"&])-([''!/+.'''
        stemmer = PorterStemmer()
        # lemmetizer = WordNetLemmatizer()

        details = details.translate(string.punctuation).lower()
        details = word_tokenize(details)
        details = [stemmer.stem(word) for word in details if not (word in stopwords.words('english') or word in ignore_chars)]
        # details = [lemmetizer.lemmatize(word) for word in details if not (word in stopwords.words('english') or word in ignore_chars)]
        details = ' '.join(details)

        details = details.replace("'", "")
        details = details.replace('.', '')
        details = details.replace('/', '')

        return details

    def __lsa__(self, food_list):
        stemmed_data = []
        for i in range(0, len(food_list)):
            details = food_list[i]
            details = self.__pre_processor__(details)
            stemmed_data.append(details)
        print('Building model...')
        transformer = TfidfVectorizer()
        tf_idf = transformer.fit_transform(stemmed_data).T

        _, S, Vt = svds(tf_idf, k=self.__get_max_k__())

        S = np.diag(S)
        food_profiles = S.dot(Vt)
        print('Food profiles created!')
        return food_profiles

    def get_food_recommendations(self, user_id, N, db_main, db_ai, fs_ai):
        # TODO: Add a check whether the user actually exists or not
        ordered_item_data_ids = []

        # TODO: Use MongoDB aggregation pipeline
        for order in db_main.orders_data.find({'user_id': user_id}, sort=[('date', -1)]).limit(self.__max_past_orders__()):
            ordered_item_data_ids.append(order.get('recipe_id'))

        ml_model = db_ai.recommenderModels.find_one({'modelName': self.model_name, 'modelVersion': self.model_version}, sort=[('createdAt', -1)])
        if ml_model is None:
            raise NoModel(self.model_name, self.model_version)

        model_id = ml_model.get('modelID')
        _model_created_at = ml_model.get('createdAt')
        ml_model = model_deserialize(fs_ai.get(model_id).read())
        food_profiles = ml_model.get('foodProfiles')
        food_ids_list = ml_model.get('foodIDsList')
        taste_profile = np.zeros(food_profiles[:, 0].shape)

        count = 0
        for item_data_id in ordered_item_data_ids:
            try:
                index = food_ids_list.index(item_data_id)
                taste_profile += food_profiles[:, index]
                count += 1
            except ValueError:
                logger('FOODPIE_RECOMMENDER', 'WARN', 'Item with item_data_id: {} does not exist!'.format(item_data_id))

        if count > 0:
            taste_profile = taste_profile/count
        #print(taste_profile)
        _scores = []
        #print(len(food_ids_list), food_profiles.shape)
        #print(food_profiles[:, len(food_ids_list)-1])
        for i in range(len(food_ids_list)-1):
            #print(food_profiles[:, i])
            similarity = cosine_sim(taste_profile, food_profiles[:, i])
            _scores.append((str(food_ids_list[i]), similarity))
        _scores = sort_tuple(data=_scores, sort_key=1, descending=True)

        _final_list = []
        for _s in _scores:
            _final_list.append({'itemDataID': _s[0], 'score': _s[1]})

        reco = {}
        reco['userID'] = user_id
        reco['foodRecommendations'] = _final_list
        reco['modelName'] = self.model_name
        reco['modelVersion'] = self.model_version
        reco['createdAt'] = datetime.datetime.utcnow()
        reco['modelCreatedAt'] = _model_created_at

        db_ai.foodRecommendations.update_one({'userID': user_id, 'modelName': self.model_name, 'modelVersion': self.model_version}, {'$set': reco}, upsert=True)
        return reco

    def update_model(self, db_main, db_ai, fs_ai):
        raw_data = self.__get_data__(db_main)
        food_ids_list, food_list = self.__get_food_content__(raw_data)
        food_profiles = self.__lsa__(food_list)

        _model = {}
        _model['foodProfiles'] = food_profiles
        _model['foodIDsList'] = food_ids_list
        model_id = fs_ai.put(model_serialize(_model))

        ml_model = {}
        ml_model['modelName'] = self.model_name
        ml_model['modelVersion'] = self.model_version
        ml_model['modelID'] = model_id
        ml_model['createdAt'] = datetime.datetime.utcnow()
        db_ai.recommenderModels.update_one({'modelName': self.model_name, 'modelVersion': self.model_version}, {'$set': ml_model}, upsert=True)
