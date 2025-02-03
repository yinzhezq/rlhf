import torch

from util import TokenizerUtil
from datasets import load_dataset
from transformers import default_data_collator
from transformers import AutoConfig, AutoModelForCausalLM, get_scheduler
import lora
from accelerate import Accelerator


def get_tokenizer():
    # 分词器
    tokenizer = TokenizerUtil()

    input_ids, _ = tokenizer.encode('how are you', max_length=6)

    input_ids, attention_mask = tokenizer.pad_to_left(input_ids)

    input_ids, attention_mask, tokenizer.decode(input_ids)

    return tokenizer


def load_data(tokenizer):
    # 加载数据
    dataset = load_dataset('json', data_files='dataset/train.json', split='train')

    # 2,4,4切分,取最后一部分
    dataset = dataset.select(range(45000, len(dataset)))

    def f(data):
        input_ids, _ = tokenizer.encode(data['prompt'], max_length=256)
        input_ids, attention_mask = tokenizer.pad_to_left(input_ids)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    dataset = dataset.map(f, remove_columns=dataset.column_names)

    loader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=default_data_collator,
                                         batch_size=4,
                                         shuffle=True,
                                         drop_last=True)

    len(loader), next(iter(loader))

    return loader


def load_model_actor():
    # 加载演员模型
    model_actor = AutoModelForCausalLM.from_pretrained('model/actor', dropout=0.0)
    lora.insert(model_actor)

    # 演员模型组件
    def f():
        params = []
        params_lora = []
        for n, p in model_actor.named_parameters():
            if not p.requires_grad:
                continue
            if 'lora_A' in n or 'lora_B' in n:
                params_lora.append(p)
                continue
            params.append(p)

        return [{
            'params': params,
            'weight_decay': 0.0
        }, {
            'params': params_lora,
            'weight_decay': 0.0,
            'lr': 5e-4
        }]

    optimizer_actor = torch.optim.Adam(f(), lr=1e-5, betas=(0.9, 0.95))

    scheduler_actor = get_scheduler(name='cosine',
                                    optimizer=optimizer_actor,
                                    num_warmup_steps=100,
                                    num_training_steps=800)

    model_actor.gradient_checkpointing_enable()
    model_actor.train()

    lora.count_params(model_actor)

    return model_actor, optimizer_actor, scheduler_actor


# 定义评委模型
class CriticModel(torch.nn.Module):

    def __init__(self, tokenizer):
        super().__init__()

        from transformers import AutoModel
        self.rwtransformer = AutoModel.from_pretrained('facebook/opt-350m', dropout=0.0)
        self.v_head = torch.nn.Linear(512, 1, bias=False)
        self.tokenizer = tokenizer

    def get_value(self, input_ids, attention_mask):
        value = self.rwtransformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.v_head(value).squeeze(2)

    def get_reward(self, input_ids, attention_mask):
        value = self.get_value(input_ids, attention_mask)

        reward = []
        for i, v in zip(input_ids, value):
            end = input_ids.shape[1] - 1
            if self.tokenizer.eos_token_id in i:
                end = i.tolist().index(self.tokenizer.eos_token_id)
            reward.append(v[end])
        reward = torch.stack(reward)

        return reward


def load_model_critic():
    # 加载评委模型
    model_critic = torch.load('model/critic')

    # 设置评委模型组件
    optimizer_critic = torch.optim.Adam(model_critic.parameters(),
                                        lr=5e-6,
                                        betas=(0.9, 0.95))

    scheduler_critic = get_scheduler(name='cosine',
                                     optimizer=optimizer_critic,
                                     num_warmup_steps=100,
                                     num_training_steps=800)

    model_critic.train()

    lora.count_params(model_critic)

    return model_critic, optimizer_critic, scheduler_critic


tokenizer = get_tokenizer()
loader = load_data(tokenizer)
model_actor, optimizer_actor, scheduler_actor = load_model_actor()
model_critic, optimizer_critic, scheduler_critic = load_model_critic()

# 加载演员参照模型
model_ref = AutoModelForCausalLM.from_pretrained('model/actor')
# 加载评委参照模型
model_reward = torch.load('model/critic')

model_ref.eval()
model_reward.eval()

accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='fp16')

(loader,
 model_actor, optimizer_actor, scheduler_actor,
 model_critic, optimizer_critic, scheduler_critic) = accelerator.prepare(
    loader, model_actor, optimizer_actor, scheduler_actor,
    model_critic, optimizer_critic, scheduler_critic)


@torch.no_grad()
def get_generate(input_ids, attention_mask):
    generate = model_actor.generate(input_ids,
                                    attention_mask=attention_mask,
                                    max_length=512,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)

    lens = (generate[:, 256:] != tokenizer.pad_token_id).sum(1)

    return generate[lens > 1]


data = next(iter(loader))
get_generate(**data).shape


def get_prob(prob, index):
    prob = prob.log_softmax(dim=2)
    prob = prob.gather(dim=2, index=index.unsqueeze(2))
    return prob.squeeze(2)


get_prob(torch.randn(4, 123, 999), torch.randint(0, 999, (4, 123))).shape
last_generate = None


@torch.no_grad()
def get_batch(input_ids, attention_mask):
    # input_ids -> [4, 256]
    # attention_mask -> [4, 256]
    global last_generate

    # 根据问题生成回答
    # [4, gen_lens]
    generate = get_generate(input_ids, attention_mask)

    # 制作缓存,防止所有回答为空的情况
    if len(generate):
        last_generate = generate
    else:
        generate = last_generate

    # [4, gen_lens]
    generate_mask = (generate != tokenizer.pad_token_id).long()

    # 两个模型分别取回答被预测到的概率
    # [4, gen_lens-1]
    prob_old = model_actor(input_ids=generate, attention_mask=generate_mask).logits
    prob_old = get_prob(prob_old[:, :-1], generate[:, 1:])

    # 取每个词的value
    # [4, gen_lens-1]
    value_old = model_critic.get_value(generate, generate_mask)[:, :-1]

    # [4, gen_lens-1]
    prob_ref = model_ref(input_ids=generate.to('cpu'), attention_mask=generate_mask.to('cpu')).logits.to('cuda')
    prob_ref = get_prob(prob_ref[:, :-1], generate[:, 1:])

    # 取回答的分数
    # [4]
    reward = model_reward.get_reward(generate.to('cpu'), generate_mask.to('cpu')).to('cuda')

    return generate, generate_mask, prob_old, prob_ref, value_old, reward


generate, generate_mask, prob_old, prob_ref, value_old, reward = get_batch(**data)
# generate.shape, generate_mask.shape, prob_old.shape, prob_ref.shape, value_old.shape, reward.shape


def get_reward_kl(end, prob_old, prob_ref, reward):
    # prob_old -> [4, gen_lens-1]
    # prob_ref -> [4, gen_lens-1]
    # reward -> [4]

    # 两份预测概率求kl散度
    # [4, gen_lens-1]
    reward_kl = -0.1 * (prob_old - prob_ref)

    # 把原本的reward加在kl散度的最后一个字上
    for i, e in enumerate(end):
        if e >= reward_kl.shape[1]:
            e = -1
        reward_kl[i, e] += reward[i].clamp(-5, 5)

    # [4, gen_lens-1]
    return reward_kl


end = generate_mask[:, 256:].sum(1) + 255
end = end.tolist()

reward_kl = get_reward_kl(end, prob_old, prob_ref, reward)
# reward_kl.shape


# 解释见get_delta_note函数
def get_delta(value_old, reward_kl):
    # value_old -> [4, gen_lens-1]
    # reward_kl -> [4, gen_lens-1]

    # gen_lens-2 -> 255
    delta = []
    for i in reversed(range(255, value_old.shape[1])):
        # [4]
        value_next = 0.0
        if i != value_old.shape[1] - 1:
            value_next = value_old[:, i + 1]

        # [4]
        d = reward_kl[:, i] + value_next - value_old[:, i]
        if len(delta):
            d += 0.95 * delta[-1]
        delta.append(d)

    # [4, gen_lens-256]
    delta = torch.stack(delta[::-1], dim=1)

    return delta


delta = get_delta(value_old, reward_kl)
# delta.shape


# get_delta函数的原理解释,注释性代码
# 数学上和get_delta函数等价,但是运行效率低
def get_delta_note(value_old, reward_kl):
    # 循环中自减会出问题,所以先clone一份再操作
    clone = value_old.clone()

    # 下一个词的value,减去当前词的value,相当于对value去基线,缩小数值方差
    # 每个词的value是相互独立的,前后词value的差,可以视为预测质量的衡量
    for i in range(255, value_old.shape[1]):
        value_next = 0.0
        if i != value_old.shape[1] - 1:
            value_next = value_old[:, i + 1]
        clone[:, i] = value_next - value_old[:, i]
    value_old = clone

    # 在value中融合reward,kl
    value_old += reward_kl

    # 蒙特卡洛采样法估计Q函数
    # 这里计算的其实就是adv
    delta = []
    for i in range(255, value_old.shape[1]):
        s = 0
        for j in range(i, value_old.shape[1]):
            s += value_old[:, j] * 0.95 ** (j - i)
        delta.append(s)

    return torch.stack(delta, dim=1)


# 测试两个函数是等价的,误差是由于计算机精度导致的
for i in range(1000):
    value_old_test = torch.randn(4, 285)
    reward_kl_test = torch.randn(4, 285)

    diff = get_delta(value_old_test, reward_kl_test) - get_delta_note(value_old_test, reward_kl_test)
    diff = diff.abs().max().item()
    assert diff < 1e-5
'test success'


def get_loss_actor(prob_new, prob_old, delta, generate_mask):
    prob_new = prob_new[:, 255:]
    prob_old = prob_old[:, 255:]
    generate_mask = generate_mask[:, 256:]

    # prob_new -> [4, gen_lens-256]
    # prob_old -> [4, gen_lens-256]
    # delta -> [4, gen_lens-256]
    # generate_mask -> [4, gen_lens-256]

    # 对数概率,求差就是求商,所以这里求的是新旧概率的变化率
    # [4, gen_lens-256]
    ratio = ((prob_new - prob_old) * generate_mask).exp()

    # delta是估计出来的去基线Q值,以变化率来缩放Q值
    # 最大化Q值,以此来寻找最优的actor
    # 裁剪,防止自举
    # [4, gen_lens-256]
    loss1 = delta * ratio
    loss2 = delta * ratio.clamp(0.8, 1.2)
    loss = torch.min(loss1, loss2) * generate_mask
    loss = loss.sum() / generate_mask.sum() / 8
    return -loss


loss_actor = get_loss_actor(prob_old, prob_old, delta, generate_mask)
# loss_actor


def get_loss_critic(value_new, value_old, delta, generate_mask):
    value_new = value_new[:, 255:]
    value_old = value_old[:, 255:]
    generate_mask = generate_mask[:, 256:]

    # value_new -> [4, gen_lens-256]
    # value_old -> [4, gen_lens-256]
    # delta -> [4, gen_lens-256]
    # generate_mask -> [4, gen_lens-256]

    # delta是估计出来的去基线Q值,加上value_old后还原为Q值
    # value_new和Q值求mse loss即可,因为value都是对Q函数的估计
    # 裁剪,防止自举
    # [4, gen_lens-256]
    loss1 = (value_new - delta - value_old) ** 2
    value_new = value_new.clamp(value_old - 0.2, value_old + 0.2)
    loss2 = (value_new - delta - value_old) ** 2

    # 求平均
    loss = torch.max(loss1, loss2) * generate_mask
    loss = loss.sum() / 2 / generate_mask.sum() / 8

    return loss


loss_critic = get_loss_critic(value_old, value_old, delta, generate_mask)
# loss_critic


def train(generate, generate_mask, prob_old, prob_ref, value_old, reward, do_step):
    # generate -> [4, gen_lens]
    # generate_mask -> [4, gen_lens]
    # prob_old -> [4, gen_lens-1]
    # prob_ref -> [4, gen_lens-1]
    # value_old -> [4, gen_lens-1]
    # reward -> [4]
    # do_step -> bool

    # 求出每句话结束的索引
    # [4]
    end = generate_mask[:, 256:].sum(1) + 255
    end = end.tolist()

    # 结束以后的value归零
    for i, e in enumerate(end):
        value_old[i, e + 1:] = 0

    with torch.no_grad():
        # 计算新旧概率的kl散度,再把reward加在最后一个字上
        # [4, gen_lens-1]
        reward_kl = get_reward_kl(end, prob_old, prob_ref, reward)

        # 估计去基线的Q值
        # [4, gen_lens-256]
        delta = get_delta(value_old, reward_kl)

    # 重新计算回答被生成的概率
    # [4, gen_lens-1]
    prob_new = model_actor(input_ids=generate, attention_mask=generate_mask).logits
    prob_new = get_prob(prob_new[:, :-1], generate[:, 1:])

    # 更新actor
    loss_actor = get_loss_actor(prob_new, prob_old, delta, generate_mask)
    accelerator.backward(loss_actor)
    if do_step:
        accelerator.clip_grad_norm_(
            [i for i in model_actor.parameters() if i.requires_grad], 1.0)
        optimizer_actor.step()
        scheduler_actor.step()
        optimizer_actor.zero_grad()

    # 重新计算每个词的value
    # [4, gen_lens-1]
    value_new = model_critic.get_value(input_ids=generate,
                                       attention_mask=generate_mask)[:, :-1]

    # 更新critic
    loss_critic = get_loss_critic(value_new, value_old, delta, generate_mask)
    accelerator.backward(loss_critic)
    if do_step:
        accelerator.clip_grad_norm_(model_critic.parameters(), 1.0)
        optimizer_critic.step()
        scheduler_critic.step()
        optimizer_critic.zero_grad()

    return loss_actor.item(), loss_critic.item()


train(generate, generate_mask, prob_old, prob_ref, value_old, reward, True)

for i, data in enumerate(loader):
    # 生成数据
    (generate, generate_mask, prob_old, prob_ref, value_old, reward) = get_batch(**data)
    do_step = (i + 1) % 8 == 0

    # 训练
    loss_actor, loss_critic = train(generate, generate_mask, prob_old,  prob_ref, value_old, reward, do_step)

    if do_step:
        lr_actor = optimizer_actor.param_groups[0]['lr']
        lr_critic = optimizer_critic.param_groups[0]['lr']
        print(i, len(loader), loss_actor, loss_critic, lr_actor, lr_critic)
        print(tokenizer.decode(generate[0, 256:]))
        print(reward[0].item())

    if i == 2500:
        break

lora.merge(model_actor)
model_actor.save_pretrained('model/rlhf')
