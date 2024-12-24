


# token의 길이와 분리 형태를 체크하기 위한 함수
def token_check(text, tokenizer, max_length=30):
    inputs = tokenizer(text, return_tensors='pt', max_length = max_length)
    inputs = {key: value.to('cuda') for key, value in inputs.items()}

    # 토큰 ID를 단어로 디코딩
    input_ids = inputs['input_ids']
    decoded_tokens = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    print("Decoded tokens:")
    print(decoded_tokens)

    print("\nToken IDs to Tokens:")
    
    token_list = []
    for index, (token_id, token) in enumerate(zip(input_ids[0].tolist(), tokens)):
        # 'Ġ' 기호 제거
        cleaned_token = token.replace('Ġ', '')
        token_list.append(cleaned_token)
        print(f"{index} Token ID: {token_id} -> Token: {cleaned_token}")
    return token_list




def model_generate(tokenizer, model, text, max_length = 30):
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    
    post_edit_outputs = model.generate(
    input_ids=inputs['input_ids'].to('cuda'),
    attention_mask=inputs['attention_mask'].to('cuda'),
    max_new_tokens=70,
    # do_sample = False  
    
    )
    
    decoded_texts = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]

    # for index, decoded_text in enumerate(decoded_texts):
    #     decoded_text = decoded_text.replace('<s>','')
    #     decoded_text = decoded_text.replace('</s>','')
    #     decoded_text = decoded_text.replace('[INST]','')
    #     decoded_text = decoded_text.split('[/INST]', -1)[-1]
    #     decoded_text = decoded_text.replace("\n", " ")
    decoded_text = decoded_text.strip()
    return decoded_text

