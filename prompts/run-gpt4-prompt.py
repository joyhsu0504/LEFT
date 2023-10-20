#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : run-gpt35-promptv4.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 04/13/2023
#
# This file is part of Project Neuro-Codex.
# Distributed under terms of the MIT license.

import re
import json
import itertools
import time
import os.path as osp
import random
import openai
import jacinle
import jacinle.io as io


def run_gpt(questions, prompts):
    query_str = '\n'.join([
        '<text>{}</text>'.format(q) for q in questions
    ])

    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4', # gpt-3.5-turbo
                temperature=0.7,
                messages=[
                    {'role': 'system', 'content': prompts['system']},
                    {'role': 'user', 'content': prompts['user'] + query_str}
                ],
                max_tokens=1024
            )
        except:
            print('Sleeping', flush=True)
            import time
            time.sleep(60)
        else:
            print('Success', flush=True)
            break
        
    return {
        'questions': questions,
        'response': response['choices'][0]['message']['content'],
        'raw_response': response
    }


def fix_parentheses(string):
    # fix the parentheses matching in the string, by adding/removing brakets.

    stack = list()
    output_string = ''
    for i in range(len(string)):
        if string[i] == '(':
            stack.append(i)
            output_string += string[i]
        elif string[i] == ')':
            if len(stack) == 0:
                pass
            else:
                output_string += string[i]
                stack.pop()
        else:
            output_string += string[i]

    for i in range(len(stack)):
        output_string += ')'
    return output_string


def main():
    parser = jacinle.JacArgumentParser()
    parser.add_argument('--dataset', type=str, default='clevr', choices=['clevr-rpms', 'clevr-puzzles', 'clevr-refexps', 'referit'])
    parser.add_argument('--questions', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--sample-size', type=int, default=100)
    args = parser.parse_args()

    assert args.output.endswith('.pkl')

    with open(args.prompt) as f:
        prompts_str = f.read()
        system_prmopt, user_prompt = prompts_str.split('----')
        prompts = {
            'system': system_prmopt.strip(),
            'user': user_prompt.strip()
        }

    print('System prompt:')
    print(prompts['system'])
    print('-' * 80)
    print('User prompt:')
    print(prompts['user'])

    if args.dataset.startswith('clevr'):
        with open(args.questions) as f:
            d = json.load(f)
        key_name = args.dataset.split('-')[1]
        questions = [this_d['question'] for this_d in d[key_name]]
        print(questions)
    elif args.dataset == 'referit':
        import pandas as pd
        df = pd.read_csv(args.questions)
        questions = df['utterance'].tolist()
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    if not osp.exists(args.output):
        # sampled_questions = random.sample(questions, args.sample_size)
        sampled_questions = questions
        batch_size = 1

        results = list()
        start_time = time.time()
        for i in range(0, len(sampled_questions), batch_size):
            print('\rProcessing {}:{} / {}, time elapsed: {:.2f}s speed: {:.2f}q/s, eta={:.2f}s'.format(
                i, i + batch_size, len(sampled_questions), time.time() - start_time, i / (time.time() - start_time), ((len(sampled_questions) - i) / (i / (time.time() - start_time)) if i > 0 else 0.0)
            ), end='')
            questions_batch = sampled_questions[i:i + batch_size]
            results.append(run_gpt(questions_batch, prompts))

        print('')

        io.set_fs_verbose()
        io.dump(args.output, {
            'questions': sampled_questions,
            'results': results
        })
    else:
        print('Output file already exists: directly loading from disk.')
        output_json = io.load(args.output)
        questions = output_json['questions']
        results = output_json['results']


if __name__ == '__main__':
    main()

