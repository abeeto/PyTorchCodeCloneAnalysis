import requests

class SlackMessenger:
    def __init__(self,token,target, name):
        self.token = token
        self.channel = target
        self.name=name
        
    def on_epoch_end(self,epoch, loss,accuracy):
        """
        a function to run on epoch end
        parameters:
            epoch (int)
            loss (float)
            accuracy (float)
        """
        
        message = f'<<DL listener>> Name:{self.name};  Epoch: {epoch}; Loss:{loss:.6f}; Accuracy: {accuracy:.4f}'
        
        data = {
                'token': self.token, 
                'channel': self.channel, 
                'as_user': True,
                'text': message
        }
        
        requests.post(url='https://slack.com/api/chat.postMessage', data=data)