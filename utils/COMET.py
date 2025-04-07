from transformers import T5ForConditionalGeneration
from transformers import T5TokenizerFast as T5Tokenizer

model_name = "svjack/comet-atomic-en"
# device = "cpu"
device = "cuda:0"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()

EFFECT_PREFIX = "What could happen after the next event?"
REACT_PREFIX = "What are your feelings after the following event?"
LINKER_1 = " then may lead to that "
LINKER_2 = " and "

def get_x(
        event: str,
        prefix: str,
        max_length = 128,
        num_beams = 2,
        do_sample = True,
        top_p = 0.95,
        top_k = 50,
        repetition_penalty = 2.5,
        length_penalty = 1.0,
        early_stopping = True
        ) -> str:
    '''
    Get `[xEffect]` or `[xReact]` from `[event]`
    '''
    prompt = "{}{}".format(prefix, event)
    encode = tokenizer(prompt, return_tensors='pt').to(device)
    answer = model.generate(encode.input_ids,
        max_length = max_length,
        num_beams = num_beams,
        do_sample = do_sample,
        top_p = top_p,
        top_k = top_k,
        repetition_penalty = repetition_penalty,
        length_penalty = length_penalty,
        early_stopping = early_stopping,)[0]
    decoded = tokenizer.decode(answer, skip_special_tokens=True)
    return decoded

def get_event_augmented_samples(event: str, **kwargs) -> str:
    '''
    Get event augmented samples from `[event]`
    '''
    xEffect = get_x(event, EFFECT_PREFIX, **kwargs)
    xReact = get_x(event, REACT_PREFIX, **kwargs)
    return event + LINKER_1 + xEffect + LINKER_2 + xReact

if __name__ == "__main__":
    print(get_event_augmented_samples("I am so happy the car broken down"))
