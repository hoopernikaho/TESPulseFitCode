# extract fit results, initial and final, from lmfit object.
# works in pandas which concats diff datatypes easily.

import pandas
import multiprocess as mp
from multiprocess import Pool
from telegram.bot import Bot
import numpy as np
import csv
import os

import heralded_pulses_analysis as hpa
import pulse_fit_powell as pfp
import trace_param as trcp
import pulse_utils as pu

def telegram_init():
    """initialises telegram messanger for updating fitting progress"""
    uid='304462759'
    token='351459283:AAH5Gyojsh69DtecyxFuCm_Rha4i8QNMSZo'
    sender_bot=Bot(token=token)
    sender_bot.sendMessage(chat_id=uid,text='testing')

def results_extr(r):
    """
    Extracts initial, final values from lmfit object.
    """
    results_summary=[]
    keys=[]
    for i,key in enumerate(r.init_values.keys()):
        results_summary.append(float(r.init_values.values()[i]))
        keys.append(key+'_init')
    for i,key in enumerate(r.best_values.keys()):
        results_summary.append(r.best_values.values()[i])
        keys.append(key+'_fitted')
    results_summary.append(float(r.redchi))
    keys.append('redchi')
    # print results_summary, keys
    return results_summary, keys

def results_packager(results_summary, keys, fname):
    """
    Concatinates fit results, filename or other flags if desired, in a pandas row.
    """
    # print results_summary, keys, fname
    r = np.array(results_summary).reshape(1,9)
    keys = np.array(keys).reshape(9,)
    p = pandas.DataFrame(r,columns=keys)
    # print np.array([fname],fmt='%s')
    q = pandas.DataFrame([fname],
                         columns=['fname'],
                         # dtype='U216'
                         )
    # print fname
    # print '{}\n'.format(np.array(q['fname']))
    return pandas.concat([q,p], axis=1)

def results_saver(outfile, results_summary, keys, fname, mcmc_flag, unequal_edges):
    """
    Append new pandas row to an outfile.
    """
    with open(outfile,'a+') as f:
        if not f.read(1):
            print 'creating new file'
            header = True
        else:
            header = False
        results_packager(results_summary, keys, fname).to_csv(f,header=header)
    f.close()

def fit_two_poolable(file, two_pulse_fit, pulse_params, height_th, sigma0):
    counter = mp.Value('i',0)
    try:
        time = pu.time_vector(file)
        signal = trcp.trace_extr(file,height_th)
        # fname = file.split('/')[-1].split('.trc')[0]
        fname = file
        # print('success')
        result = pfp.fit_two_cw(time,signal,
                      two_pulse_fit,
                      pulse_params,
                      height_th,
                      sigma0)
        # Extract results
        results_summary, keys = results_extr(result)
        df = results_packager(results_summary, keys, fname)
        # Sends Telegram Message to Update Status
        # counter.value+=1
        # times_toupdate = 20
        # text_tosend = '\nProcess finished:'+ '%.1f'%(counter.value/len(tasks)*100) + "%"
        # print text_tosend
        # try:
        #     if counter.value % int(len(tasks)/times_toupdate) == 0:
        #         sender_bot.sendMessage(chat_id=uid,text=text_tosend)
        # except:
        #     pass
        
        # Extracts useful data for saving
        return df
    except:
        print fname+' failed' #if file is corrupt
        raise

def run_fit_two_poolable(outfile,cores,tasks,
    two_pulse_fit, pulse_params, height_th, sigma0):
    """
    Append new pandas row to an outfile.
    """
    p = Pool(cores)
    file_exists = os.path.isfile(outfile)
    with open(outfile,'w+') as f:
        if not file_exists:
            """create header"""
            df = p.map(
                lambda f: fit_two_poolable(f,two_pulse_fit, pulse_params, height_th, sigma0),
                tasks[0:1])
            df[0].to_csv(f,header=True)
        for df in p.imap(
            lambda f: fit_two_poolable(f,two_pulse_fit, pulse_params, height_th, sigma0),
            tasks):
            try:
                df.to_csv(f,header=False)
            except:
                break