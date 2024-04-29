#%%
from botframework.connector.connector_client import ConnectorClient
from botframework.connector.models import ConversationParameters
from botframework.connector.auth.microsoft_app_credentials import MicrosoftAppCredentials
from botbuilder.core import  MessageFactory
from botbuilder.schema import ChannelAccount
from shareplum import Office365, Site
from shareplum.site import Version
import io
import pandas as pd

def call_for_content():
    APP_ID = '5995b22a-e094-4d5b-8523-edd7bb089d89'
    APP_PASSWORD = '0e.x0-WG49aoOhlm4Z2D5wE0KODRI_~G-8'
    CHANNEL_ID = 'msteams'
    BOT_ID = '5995b22a-e094-4d5b-8523-edd7bb089d89'
    authcookie = Office365('https://thespurgroup.sharepoint.com', username='kevin.lin@thespurgroup.com', password='zcmzlpbxcvtzqwzp').GetCookies()
    site = Site('https://thespurgroup.sharepoint.com/sites/bot_project_test/', version=Version.v2016, authcookie=authcookie)
    folder_path = 'Shared Documents/Lono2docs/'
    teams_tracker = folder.get_file('testing.csv')
    source_stream = io.BytesIO(teams_tracker)
    df = pd.read_csv(source_stream).drop(columns=['Unnamed: 0'])
    for index, row in df.iterrows():
        SERVICE_URL = row.serviceURL
        recipient_id = row.recipientID
        TENANT_ID = row.tenantID
        NAME = row.Name
        to = ChannelAccount(id=recipient_id)
        bot_channel = ChannelAccount(id=CHANNEL_ID)
        MicrosoftAppCredentials.trust_service_url(SERVICE_URL)
        credentials = MicrosoftAppCredentials(APP_ID, APP_PASSWORD)
        conn_client = ConnectorClient(credentials, SERVICE_URL)
        message_activity = MessageFactory.text(f"Hello {NAME}, are you going to submit an article for this month's newsletter?");
        
        conversation_params = ConversationParameters(members=[to], channel_data={ 'tenant': { 'id': TENANT_ID } })
        conversation = conn_client.conversations.create_conversation(conversation_params)
        conn_client.conversations.send_to_conversation(conversation.id, message_activity)


