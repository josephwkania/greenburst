#!/usr/bin/env python3.6

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import glob
import pika
import pandas as pd
from pika_send import send2Q
from plot_cand import plotem
import dropbox
import yaml
import re
from slack_send import send_img_2_slack

logger = logging.getLogger()
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.getLogger('pika').setLevel(logging.INFO)

__author__='Devansh Agarwal'
__email__ = 'da0017@mix.wvu.edu'

def stage_initer(values):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    
    channel.queue_declare(queue='stage03_queue', durable=True)
    
    def callback(ch, method, properties, body):
        values.file=body.decode()
        logging.info(f'got it {values.file}')
        begin_main(values)
        ch.basic_ack(delivery_tag = method.delivery_tag)
        logging.info("Ack'ed")
    
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_message_callback=callback, queue='stage03_queue')
    
    channel.start_consuming()

def begin_main(values):
    with open("config/conf.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)
    TOKEN = data_loaded['dropbox']['token']
    dbx = dropbox.Dropbox(TOKEN)
    try:
        fout, known_src = plotem(values.file, nrby=True) #modified: get nearby source info
        file_name=fout.split('/')[-1][:-3]
        with open(fout, 'rb') as f:
            data = f.read()
            response=dbx.files_upload(data,f'/plots/{file_name}png',mode=dropbox.files.WriteMode.overwrite)
            logging.info(response)
            link=dbx.sharing_create_shared_link(f'/plots/{file_name}png')
            url=link.url
            slack_send_url=re.sub(r"\&dl\=0", "&dl=1", url) # Replaced ? with & b/c change in dropbox url
        logging.info(f'Create png at {fout}')
        logging.info(f'Dropbox URL {slack_send_url}')
        logging.info(send_img_2_slack(slack_send_url, nrby=known_src))
    except Exception as error:
        logging.warning(f"Encountered {error} while processing file {values.file}")
    return None


if __name__ == '__main__':
    parser=ArgumentParser(description='Stage 3: Get Real Cands', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument('-d', '--daemon', dest='daemon', action='store_false', help='Run with AMQP')
    parser.add_argument('-f', '--file', nargs='+')
    parser.set_defaults(verbose=False)
    parser.set_defaults(daemon=True)
    values = parser.parse_args()

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)

    if values.daemon:
        logging.info('Running in daemon mode')
        stage_initer(values)
    else:
        print(values.file)
        begin_main(values)
