from albert_qna_model import run_prediction
import time

if __name__ == '__main__':

    context = "New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean. It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million. New Zealand's capital city is Wellington, and its most populous city is Auckland."
    questions = ["What is New Zealand",
                 "Where is New Zealand",
                 "How big is New Zealand",
                 "How many people live in New Zealand",
                 "What's the largest city"]

    # Run method
    start = time.time()
    predictions = run_prediction(questions, context)
    end = time.time()
    duration = end - start
    print(f"Time taken to process QnA is {duration} seconds")

    # Print results
    for key in predictions.keys():
        print(predictions[key])

    """Base model - 5 seconds
    New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean.
    southwestern Pacific Ocean
    268,000 square kilometres
    4.9 million.
    Auckland
    """

    """ Large Model - 11 Seconds
    a sovereign island country in the southwestern Pacific Ocean
    New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean.
    268,000 square kilometres (103,500 sq mi),
    4.9 million.
    Auckland
    """

    """ xxlarge - 41 Seconds
    New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean.
    New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean.
    It has a total land area of 268,000 square kilometres (103,500 sq mi),
    4.9 million.
    New Zealand's capital city is Wellington, and its most populous city is Auckland.
    
    """