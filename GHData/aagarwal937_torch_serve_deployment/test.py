# import requests
# import json
# from scipy import spatial


# queries = ['positive', 'negetive']
# # queries = [['does dyno have a slowmode command?', 'is there a slowmode command for dyno?'], ['does dyno have a slowmode command?', 'Is there a slowmode command'], ['does dyno have a slowmode command?', 'Is there a slowmode command'], ['does dyno have a slowmode command?', 'Is there a slowmode command'], ['does dyno have a slowmode command?', 'Is there a slowmode command'], ['does dyno have a slowmode command?', 'Is there a slowmode command'], ['does dyno have a slowmode command?', 'Is there a slowmode command'], ['does dyno have a slowmode command?', 'Is there a slowmode command'], ['does dyno have a slowmode command?', 'Is there a slowmode command'], ['does dyno have a slowmode command?', 'Is there a slowmode command']]
# input = {'queries':queries}

# response = requests.post('http://0.0.0.0:8080/predictions/bert', data={'data':json.dumps(input)})
# # print(response, response.status_code, response.text, response.headers)

# data  = response.content
# print(data)
# # if response.status_code == 200:
# #     vectors = response.json()
# #     similarity = 1- spatial.distance.cosine(vectors[0], vectors[1])
# #     print(round(similarity, 3))



# importing standard modules ==================================================
from typing import List, Dict
import json, datetime


# importing third-party modules ===============================================
import requests
from scipy import spatial


# importing custom modules ====================================================


# module variables ============================================================
INFERENCE_API_HOST: str = "http://localhost:9090"
INFERENCE_MODEL_NAME: str = "bert"


# method definitions ==========================================================
def run_main() -> None:
    __ts1__ =  datetime.datetime.now()
    print(
        "{:<80}; timestamp: {:>30}".format("started 'run_main'", str(__ts1__))
    )

    template_url: str = "{}/predictions/{}"
    url: str = template_url.format(
        INFERENCE_API_HOST, INFERENCE_MODEL_NAME
    )

    queries: List[str] = [
        'does dyno have a slowmode command?', 
        'is there a slowmode command for dyno?'
    ]

    try:
        __ts2__ = datetime.datetime.now()
        print("{:<80}; timestamp: {:>30}; diff: {:>20}"\
            .format(
                "sent request to '{}'".format(url), 
                str(__ts2__), str(__ts2__ - __ts1__))
            )


        response: requests.Response = requests.post(
            url, 
            data={
                "data": json.dumps({
                    "queries": queries
                })
            }
        )


        __ts3__ =  datetime.datetime.now()
        print("{:<80}; timestamp: {:>30}; diff: {:>20}"\
            .format("received response", str(__ts3__), str(__ts3__ - __ts2__)))


        response.raise_for_status()
        vectors: List = response.json()

    except Exception as error:
        print("ERROR: {} - {}".format(type(error), str(error)))
        raise

    # if network code all cool
    similarity_score: float = 1 - spatial.distance.cosine(vectors[0], vectors[1])

    __ts4__ = datetime.datetime.now()
    print("{:<80}; timestamp: {:>30}; diff: {:>20}"\
        .format("calculated similarity", str(__ts4__), str(__ts4__ - __ts3__)))

    __ts5__ = datetime.datetime.now()
    print("{:<80}; timestamp: {:>30}; diff: {:>20}"\
        .format("finished test case", str(__ts5__), str(__ts5__ - __ts1__)))
    
    print("===================================================================="
    "\nSimilarity score for\n{} = {}".format(queries, similarity_score))

    return None


# main ========================================================================
if __name__ == "__main__":
    run_main()