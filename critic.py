import torch

from util import TokenizerUtil
from datasets import load_dataset
from lora import count_params
from transformers import get_scheduler
from accelerate import Accelerator

tokenizer = TokenizerUtil()

input_ids, attention_mask = tokenizer.encode('how are you', max_length=6)

input_ids, attention_mask, tokenizer.decode(input_ids)

dataset = load_dataset('json', data_files='dataset/train.json', split='train')

# 2,4,4切分,取第1部分
dataset = dataset.select(range(15000, 45000))


def f(data):
    # 区分两种生成结果
    chosen = data['prompt'] + data['chosen'].swapcase()
    rejected = data['prompt'] + data['chosen']

    chosen_input_ids, chosen_attention_mask = tokenizer.encode(chosen)
    rejected_input_ids, rejected_attention_mask = tokenizer.encode(rejected)

    return {
        'chosen_input_ids': chosen_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'rejected_input_ids': rejected_input_ids,
        'rejected_attention_mask': rejected_attention_mask
    }


dataset = dataset.map(f)
dataset.set_format('torch')


def f(data):
    chosen_input_ids = [i['chosen_input_ids'] for i in data]
    chosen_attention_mask = [i['chosen_attention_mask'] for i in data]
    rejected_input_ids = [i['rejected_input_ids'] for i in data]
    rejected_attention_mask = [i['rejected_attention_mask'] for i in data]

    input_ids = torch.stack(chosen_input_ids + rejected_input_ids, dim=0)
    attention_mask = torch.stack(chosen_attention_mask +
                                 rejected_attention_mask,
                                 dim=0)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


loader = torch.utils.data.DataLoader(dataset,
                                     collate_fn=f,
                                     batch_size=4,
                                     shuffle=True,
                                     drop_last=True)

len(loader), next(iter(loader))


class CriticModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        from transformers import AutoModel
        self.rwtransformer = AutoModel.from_pretrained('facebook/opt-350m',
                                                       dropout=0.0)

        self.v_head = torch.nn.Linear(512, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        value = self.rwtransformer(
            input_ids=input_ids,
            attention_mask=attention_mask).last_hidden_state
        value = self.v_head(value).squeeze(-1)

        loss_sum = 0.0
        value_chosen_sum = 0.0
        value_rejected_sum = 0.0
        for input_ids_chosen, input_ids_rejected, value_chosen, value_rejected in zip(
                input_ids[:4], input_ids[4:], value[:4], value[4:]):
            # 找出每条回答中的起止索引
            start = (
                    input_ids_chosen == input_ids_rejected).tolist().index(False)

            end_chosen = input_ids_chosen.tolist().index(
                tokenizer.eos_token_id) + 1
            end_rejected = input_ids_rejected.tolist().index(
                tokenizer.eos_token_id) + 1
            end = max(end_chosen, end_rejected)

            value_chosen = value_chosen[start:end]
            value_rejected = value_rejected[start:end]

            loss = value_chosen - value_rejected
            loss = -torch.nn.functional.logsigmoid(loss).mean()

            loss_sum += loss
            value_chosen_sum += value_chosen.mean().item()
            value_rejected_sum += value_rejected.mean().item()

        return loss_sum / 4, value_chosen_sum, value_rejected_sum


model_critic = CriticModel()

count_params(model_critic)


def f():
    params_decay = []
    params = []
    for name, param in model_critic.named_parameters():
        if 'bias' in name or 'norm.weight' in name:
            params.append(param)
            continue
        params_decay.append(param)

    return [{
        'params': params_decay,
        'weight_decay': 0.1
    }, {
        'params': params,
        'weight_decay': 0.0
    }]


optimizer = torch.optim.Adam(f(), lr=5e-5, betas=(0.9, 0.95))

scheduler = get_scheduler(name='cosine',
                          optimizer=optimizer,
                          num_warmup_steps=0,
                          num_training_steps=500)

accelerator = Accelerator(gradient_accumulation_steps=16,
                          mixed_precision='fp16')

model_critic, loader, optimizer, scheduler = accelerator.prepare(model_critic, loader, optimizer, scheduler)

model_critic.train()

for i, data in enumerate(loader):
    with accelerator.accumulate(model_critic):
        loss, value_chosen_sum, value_rejected_sum = model_critic(**data)
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model_critic.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if (i + 1) % 100 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(i, len(loader), loss.item(), lr, value_chosen_sum, value_rejected_sum)

    if i == 2000:
        break

torch.save(model_critic.to('cpu'), 'model/critic')
