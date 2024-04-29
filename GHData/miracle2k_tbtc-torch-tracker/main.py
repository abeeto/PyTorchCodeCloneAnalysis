#!/usr/bin/env python
import time
import json
import asyncio
import aiohttp
from aiohttp_requests import requests
import calendar
from dateutil.parser import parse
from quart_cors import cors


initial = "0xb6f512124d8ddb8d7ef8559318de92531af32f6f"
chain = []

async def watch_current():
    sleep_amount = 3

    while True:
        prev = chain[-1] if chain else None
        current = prev['to'] if prev else initial

        print('check ' + current)
        try:
            response = await asyncio.wait_for(
                requests.get(f"https://api.zksync.io/api/v0.1/account/{current}/history/0/15"), 
                timeout=15.0)
        except asyncio.TimeoutError:
            print('timeout!')
            await asyncio.sleep(2)
            continue

        data = await response.json()
        
        for tx in data:            
            #print(json.dumps(tx, indent=4))
            created_at = calendar.timegm(parse(tx['created_at']).timetuple())
            if tx['tx']['type'] == 'Transfer' and tx['tx']['from'] == current and tx['tx']['token'] == 'TBTC' and (not prev or created_at > prev['timestamp']):
                print("found tx -> " + tx['tx']["to"])
                chain.append({
                    "from": tx['tx']["from"],
                    "to": tx['tx']["to"],
                    "amount": tx['tx']["amount"],
                    "fee": tx['tx']["fee"],
                    "tx_id": tx['tx_id'],
                    "date": tx['created_at'],
                    "timestamp": created_at,
                })
                sleep_amount = 3
                break
        else:
            sleep_amount = 60

        await asyncio.sleep(sleep_amount)



from quart import Quart

app = Quart(__name__)
app = cors(app, allow_origin="*")



@app.before_serving
async def create_db_pool():
    future = asyncio.ensure_future(watch_current())
    def stop(f):
        # TOOD: Not the right way to stop, show a warning. what is?
        if f.exception():
            asyncio.get_running_loop().stop()
    future.add_done_callback(stop)


@app.route('/torch')
async def torch():
    return json.dumps(chain)


if __name__ == '__main__':
    app.run()
