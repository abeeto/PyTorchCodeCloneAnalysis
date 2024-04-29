from scene_recog_service import predict
import uuid
from db_models.mongo_setup import global_init
from db_models.models.cache_model import Cache
import init
import globals
import requests
from init import ERR_LOGGER
import os

global_init()

FILE_ID = ""


def save_to_db(db_object, labels, scores):
    try:
        print("*****************SAVING TO DB******************************")
        print("in save")
        print(db_object)
        print(db_object.id)
        db_object.labels = labels
        db_object.scores = scores
        db_object.save()
        print("*****************SAVED TO DB******************************")
    except Exception as e:
        print(f"{e} ERROR IN SAVE TO DB FILE ID {FILE_ID}")
        ERR_LOGGER(f"{e} ERROR IN SAVE TO DB FILE ID {FILE_ID}")


def update_state(file_name):
    payload = {
        'parent_name': globals.PARENT_NAME,
        'group_name': globals.GROUP_NAME,
        'container_name': globals.RECEIVE_TOPIC,
        'file_name': file_name,
        'client_id': globals.CLIENT_ID
    }
    try:
        requests.request("POST", globals.DASHBOARD_URL, data=payload)
    except Exception as e:
        print(f"{e} EXCEPTION IN UPDATE STATE API CALL......")
        ERR_LOGGER(f"{e} EXCEPTION IN UPDATE STATE API CALL......FILE ID {FILE_ID}")


if __name__ == "__main__":
    print("Connected to Kafka at " + globals.KAFKA_HOSTNAME + ":" + globals.KAFKA_PORT)
    print("Kafka Consumer topic for this Container is " + globals.RECEIVE_TOPIC)
    for message in init.consumer_obj:
        message = message.value
        db_key = str(message)
        print(db_key, 'db_key')
        try:
            db_object = Cache.objects.get(pk=db_key)
        except Exception as e:
            print(f"{e} EXCEPTION IN GET PK... continue")
            ERR_LOGGER(f"{e} EXCEPTION IN GET PK... continue")
            continue

        file_name = db_object.file_name

        final_labels = db_object.labels
        final_scores = db_object.scores

        print("#############################################")
        print("########## PROCESSING FILE " + file_name)
        print("#############################################")

        if db_object.is_doc_type:
            """document"""
            if db_object.contains_images:
                images_array = []
                for image in db_object.files:
                    pdf_image = str(uuid.uuid4()) + ".jpg"
                    with open(pdf_image, 'wb') as file_to_save:
                        file_to_save.write(image.file.read())
                    images_array.append(pdf_image)

                for image in images_array:

                    try:
                        response = predict(file_name=image)
                    except Exception as e:
                        print(f"{e} ERROR IN PREDICT")
                        ERR_LOGGER(f"{e} Exception in predict FILE ID {FILE_ID}")
                        os.remove(image)
                        continue
                    # final_labels.extend(response["labels"])
                    for label, score in zip(response["labels"], response['scores']):
                        if label not in final_labels:
                            final_labels.append(label.strip())
                            final_scores.append(score)
                        else:
                            x = final_labels.index(label)
                            score_to_check = final_scores[x]
                            if score > score_to_check:
                                final_scores[x] = score

                save_to_db(db_object, final_labels, final_scores)
                print(".....................FINISHED PROCESSING FILE.....................")
                update_state(file_name)
            else:
                pass
        else:
            """image"""
            with open(file_name, 'wb') as file_to_save:
                file_to_save.write(db_object.file.read())
            try:
                response = predict(file_name)
            except Exception as e:
                print("ERROR IN PREDICE")
                ERR_LOGGER(f"{e} Exception in predict FILE ID {FILE_ID}")
                os.remove(file_name)
                continue

            for label, score in zip(response["labels"], response['scores']):
                if label not in final_labels:
                    final_labels.append(label.strip())
                    final_scores.append(score)
                else:
                    x = final_labels.index(label)
                    score_to_check = final_scores[x]
                    if score > score_to_check:
                        final_scores[x] = score

            print("to_save audio", final_labels, final_scores)
            save_to_db(db_object, final_labels, final_scores)
            print(".....................FINISHED PROCESSING FILE.....................")
            update_state(file_name)