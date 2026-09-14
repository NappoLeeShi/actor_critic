# REWRITE PROJECT: CORRECT ONE-STEP ACTOR–CRITIC FOR 4×4 WUMPUS WORLD

Bạn đang làm việc trên một project Actor–Critic reinforcement learning bằng **NumPy thuần** cho Wumpus World 4×4. Hãy **kiểm tra toàn bộ code hiện tại, sửa implementation, chạy lại experiment, rồi cập nhật report theo đúng kết quả thực tế**.

## 1. Mục tiêu chính

Mục tiêu của project là minh họa rõ và đúng cơ chế:

$$
State
\rightarrow Actor
\rightarrow Action
\rightarrow Environment
\rightarrow Reward + Next\ State
\rightarrow Critic
\rightarrow TD\ Target
\rightarrow TD\ Error
\rightarrow Actor/Critic\ Update
$$

Đây phải là **One-Step Actor–Critic**, không gọi là A2C nếu implementation vẫn là một update sau mỗi environment step.

Không được chỉ sửa report để che lỗi. Phải sửa code, chạy lại và dùng kết quả thật.

---

# 2. Giữ yêu cầu NumPy-only

Không dùng:

* PyTorch
* TensorFlow
* Keras
* sklearn
* automatic differentiation
* Gym dependency

Cho phép:

* NumPy
* matplotlib
* Python standard library

Backpropagation phải được viết thủ công.

---

# 3. Environment: Wumpus World 4×4

Giữ grid 4×4.

Mỗi episode:

* 1 Agent
* 1 Gold
* 1 Wumpus
* 2–4 Pits
* Agent bắt đầu ở một vị trí cố định, ví dụ (0,0)
* Gold/Wumpus/Pits được random nhưng không được đặt lên ô start

Actions:

```text
0 = Up
1 = Down
2 = Left
3 = Right
```

Môi trường deterministic:

Nếu Agent ở `(x,y)` và chọn Right thì:

```text
(x,y) → (x+1,y)
```

nếu không ra ngoài grid.

Transition probability trong môi trường này về bản chất là deterministic:

$$
P(s'|s,a)\in\{0,1\}
$$

Nếu action hợp lệ:

$$
P(s'|s,a)=1
$$

không cần tạo randomness trong movement.

Nếu action đẩy Agent ra ngoài grid:

* Agent giữ nguyên vị trí
* nhận normal move reward

Episode kết thúc khi:

* Agent tới Gold
* Agent tới Pit
* Agent tới Wumpus
* vượt quá `MAX_STEPS`

Reward môi trường:

```text
normal move = -1
gold        = +100
pit         = -100
wumpus      = -100
```

Có thể scale reward khi training, nhưng phải giải thích rõ raw reward và training reward khác nhau.

---

# 4. Sửa State Representation

Đây là thay đổi quan trọng.

State hiện tại chỉ:

```text
[breeze, stench, glitter, x, y]
```

không đủ để Agent phân biệt các random maps có cùng vị trí nhưng Gold/Pit/Wumpus khác nhau.

Hãy thiết kế state theo kiểu **fully observable Markov state** để case study train nhiều map và test map mới có ý nghĩa.

Đề xuất:

### Input = toàn bộ 4×4 map được encode bằng các channel

Mỗi cell có:

* Agent
* Gold
* Wumpus
* Pit

Ví dụ 4 channels × 16 cells:

$$
4\times4\times4=64
$$

Flatten thành vector 64 chiều.

Có thể bổ sung:

* current agent position
* breeze
* stench

nếu hữu ích, nhưng không được tạo redundancy vô lý.

Quan trọng:

> State phải đủ thông tin để cùng một state representation xác định được đúng tình huống hiện tại.

Mục tiêu là:

```text
Train:
random Map 1
random Map 2
...
random Map N

Test:
Map chưa từng xuất hiện
```

Agent phải học policy tổng quát thay vì học thuộc một vị trí Gold cố định.

Trong report phải nói rõ đây là **fully observable Wumpus variant**, không phải partially observable classic Wumpus.

---

# 5. Actor Network

Dùng NumPy MLP:

```text
Input
 ↓
Linear(64 → 128)
 ↓
ReLU
 ↓
Linear(128 → 64)
 ↓
ReLU
 ↓
Linear(64 → 4)
 ↓
logits
```

Không đặt Softmax trực tiếp vào lớp Linear cuối nếu code hiện tại đang làm sampling từ logits.

Khi chọn action:

$$
\pi_\theta(a|s)=softmax(logits)
$$

Dùng numerically stable softmax:

$$
p_i=
\frac{e^{z_i-\max(z)}}
{\sum_j e^{z_j-\max(z)}}
$$

Training:

```text
sample action from categorical distribution
```

Không dùng argmax trong training.

Sau training:

```text
argmax(probabilities)
```

để tạo deterministic policy.

Actor parameter gồm:

```text
weights + biases
```

và các parameter này phải được giữ lại giữa các episode.

---

# 6. Critic Network

Dùng:

```text
Input
 ↓
Linear(64 → 128)
 ↓
ReLU
 ↓
Linear(128 → 64)
 ↓
ReLU
 ↓
Linear(64 → 1)
 ↓
V(s)
```

Output cuối là **scalar linear output**:

$$
V_\phi(s)
$$

Không dùng Softmax cho Critic.

---

# 7. ReLU

Hidden layers dùng:

$$
ReLU(x)=\max(0,x)
$$

Phải implement thủ công bằng NumPy:

```python
np.maximum(z, 0)
```

Trong backprop:

$$
ReLU'(z)=
\begin{cases}
1,&z>0\\
0,&z\le0
\end{cases}
$$

---

# 8. One-Step TD Learning

Với transition:

$$
(s_t,a_t,r_t,s_{t+1},done)
$$

Critic tính:

$$
V_t=V_\phi(s_t)
$$

Nếu terminal:

$$
V_{t+1}=0
$$

Nếu không terminal:

$$
V_{t+1}=V_\phi(s_{t+1})
$$

TD Target:

$$
\boxed{
y_t=r_t+\gamma(1-done)V_\phi(s_{t+1})
}
$$

TD Error:

$$
\boxed{
\delta_t=y_t-V_\phi(s_t)
}
$$

Trong one-step Actor–Critic:

$$
\boxed{
\delta_t \approx A(s_t,a_t)
}
$$

Phải giải thích rõ:

$$
A(s,a)=Q(s,a)-V(s)
$$

là Advantage thật về mặt định nghĩa.

TD Error **không đồng nhất về định nghĩa với Advantage**, mà được dùng như một **one-step estimator của Advantage**.

Không train Q-network trong implementation này.

---

# 9. CRITICAL BUG: sửa Critic gradient

Nếu dùng:

$$
L_{critic}
=
\frac12(y_t-V_\phi(s_t))^2
$$

thì:

$$
\frac{\partial L}{\partial V}
=
V-y
=
-\delta
$$

Do đó backpropagation của Critic phải dùng đúng gradient:

$$
\boxed{
\frac{\partial L}{\partial V}=-\delta
}
$$

Không được truyền `+delta` vào backward nếu optimizer đang thực hiện:

$$
\theta\leftarrow\theta-\eta\nabla_\theta L
$$

Hãy viết unit test nhỏ để kiểm tra hướng update của Critic.

Test case:

```text
prediction = 2
target = 8
```

Critic phải được update theo hướng làm:

```text
prediction tăng về phía 8
```

không được giảm xuống.

---

# 10. Critic Loss

Dùng:

$$
\boxed{
L_{critic}
=
\frac12(y_t-V_\phi(s_t))^2
}
$$

Không dùng Advantage như thể Advantage tự nó là loss.

Có thể đặt:

```python
td_error = target - value
critic_loss = 0.5 * td_error**2
```

Sau đó backprop với:

```python
grad_output = -td_error
```

hoặc cách tương đương về toán học.

---

# 11. Actor Loss

Dùng policy-gradient estimator:

$$
\boxed{
L_{actor}
=
-\log \pi_\theta(a_t|s_t)\delta_t
}
$$

Gradient đối với logits:

$$
\boxed{
\frac{\partial L}{\partial logits}
=
-\delta_t(p-onehot(a_t))
}
$$

Kiểm tra kỹ dấu gradient.

Nếu:

$$
\delta>0
$$

thì action vừa chọn phải được reinforce.

Nếu:

$$
\delta<0
$$

thì action vừa chọn phải bị giảm xác suất.

---

# 12. Exploration

Training:

```text
sample from softmax policy
```

không dùng greedy.

Có thể thêm entropy bonus để tránh policy collapse quá sớm:

$$
H(\pi)=-\sum_a p(a)\log p(a)
$$

Nếu dùng entropy coefficient, phải implement gradient đúng và ghi rõ trong report.

Có thể dùng:

```text
ENTROPY_COEF = 0.01 ~ 0.05
```

nhưng hãy tune dựa trên experiment, không tự tuyên bố một giá trị là tối ưu.

---

# 13. Optimizer

Không được ghi "plain SGD, no momentum" nếu code vẫn dùng velocity accumulation.

Chọn **một trong hai**, và phải nhất quán:

### Preferred for this educational project:

Vanilla SGD

$$
W\leftarrow W-\eta\nabla_W L
$$

$$
b\leftarrow b-\eta\nabla_b L
$$

Không velocity, không momentum.

Learning rate Actor và Critic có thể tách riêng.

Ví dụ:

```text
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
```

Nếu cần tune thì thử một vài giá trị và ghi rõ kết quả thực nghiệm.

Không gọi optimizer là Adam nếu code không implement Adam.

---

# 14. Backpropagation phải được giải thích rõ

Implement:

```text
forward
→ cache z, activation, input
→ loss
→ backward from output to input
→ gradients dW/db
→ parameter update
```

Phải có code:

```python
dW = outer(gradient, input)
db = gradient
gradient_to_previous = W.T @ gradient
```

và ReLU derivative:

```python
gradient *= (z > 0)
```

Hãy viết test gradient đơn giản bằng finite differences nếu có thể.

---

# 15. Training Setup

Train trên nhiều random maps.

Ví dụ:

```text
EPISODES = 5000
```

hoặc một số lượng phù hợp nếu 2000 chưa đủ.

Mỗi episode tạo map mới.

Không reset Actor/Critic weights giữa các episode.

Pseudo-code:

```python
initialize_actor()
initialize_critic()

for episode in range(EPISODES):

    env.reset_random_map()
    state = env.get_state()

    while not done:

        logits = actor.forward(state)
        probs = softmax(logits)

        action = sample(probs)

        next_state, reward, done, info = env.step(action)

        value = critic.forward(state)

        next_value = 0 if done else critic.forward(next_state)

        target = scaled_reward + gamma * next_value

        td_error = target - value

        update_critic(td_error)

        update_actor(td_error)

        state = next_state
```

---

# 16. Train/Test Split phải rõ ràng

Không được đánh giá bằng cùng những map mà Agent đã train.

Training:

```text
random seeds / training maps
```

Testing:

```text
different fixed seeds
```

Ví dụ:

```text
Training seeds:
0–9999

Test seeds:
10000–10199
```

Test maps phải được tạo riêng sau khi training hoàn tất.

---

# 17. Metrics cần báo cáo

Không chỉ dùng reward.

Bắt buộc có:

### 1. Average episode return

$$
R=\sum_t r_t
$$

### 2. Success rate

$$
Success\ Rate=
\frac{\text{gold wins}}
{\text{test episodes}}
$$

### 3. Hazard rate

Tỷ lệ chết bởi:

* Pit
* Wumpus

### 4. Timeout rate

### 5. Average steps

Tách:

```text
average steps on successful episodes
```

và có thể báo thêm overall average steps.

### 6. Training curve

Vẽ:

```text
episode vs return
moving average
```

### 7. Test performance

Báo riêng:

```text
Train performance
Test on unseen maps
```

Không gộp hai cái thành một con số.

---

# 18. Visualization

Tạo ít nhất:

### Plot 1

Episode return + moving average.

### Plot 2

Success rate theo training progress.

### Plot 3

Final policy trên một test map.

Ví dụ:

```text
↑ → → G
↑ P W →
↑ → → →
A → → →
```

### Plot 4

Value map:

```text
V(s)
```

trên một test map để cho thấy state gần Gold có xu hướng value tốt hơn và state nguy hiểm có value thấp hơn.

Nếu visualization policy được tạo bằng greedy argmax:

```python
action = np.argmax(probs)
```

phải nói rõ đây là **deterministic policy sau training**.

---

# 19. Kiểm tra hành vi Agent

Sau training, chạy ít nhất 10–50 test episodes trên unseen maps và lưu:

```text
map
action sequence
reward
terminal outcome
number of steps
```

In ra một vài episode dễ hiểu.

Ví dụ:

```text
Episode:
Start
↓
Right
↓
Right
↓
Up
↓
Gold

Total reward = ...
Steps = ...
Outcome = GOLD
```

Nếu agent thất bại, cũng phải ghi nhận thất bại thật, không sửa output.

---

# 20. Không được "fake improvement"

Đây là yêu cầu bắt buộc.

Không được:

* sửa số liệu bằng tay
* tự tạo success rate
* viết report là model tốt khi code thực tế không tốt
* cherry-pick một run đẹp duy nhất
* gọi training là convergence chỉ vì curve phẳng

Nếu model vẫn underperform:

> ghi rõ underperform và phân tích nguyên nhân.

---

# 21. Rewrite report

Sau khi code được sửa và chạy thành công, rewrite report theo đúng implementation mới.

### Đổi tiêu đề:

```text
One-Step Actor–Critic in a 4×4 Wumpus World
```

không gọi là A2C nếu implementation không phải A2C chuẩn.

### Report phải giải thích rõ:

#### Actor

$$
\pi_\theta(a|s)
$$

Actor chọn action.

#### Critic

$$
V_\phi(s)
$$

Critic đánh giá state.

#### TD Target

$$
y_t=r_t+\gamma V(s_{t+1})
$$

TD Target không phải prediction của state tiếp theo.

Nó là:

```text
reward hiện tại
+
discounted value prediction của next state
```

#### TD Error

$$
\delta_t=y_t-V(s_t)
$$

#### Advantage

$$
A(s,a)=Q(s,a)-V(s)
$$

và trong one-step Actor-Critic:

$$
\delta_t\approx A(s_t,a_t)
$$

#### Neural Network

Giải thích:

```text
weights
biases
linear layer
ReLU
logits
softmax
forward pass
loss
backpropagation
gradient
parameter update
```

---

# 22. Report phải phân biệt parameter và hyperparameter

### Hyperparameters

Ví dụ:

```text
learning rate
gamma
number of layers
hidden size
activation
entropy coefficient
number of episodes
max steps
```

### Parameters

```text
weights
biases
```

Weights/biases được learning bằng gradient descent.

---

# 23. Report phải giải thích Markov Property

Với state representation mới:

> State phải chứa đủ thông tin để transition và expected future reward phụ thuộc vào state hiện tại + action, thay vì cần toàn bộ lịch sử.

Nói đơn giản:

$$
P(s_{t+1}|s_t,a_t,\text{history})
=
P(s_{t+1}|s_t,a_t)
$$

Trong environment deterministic:

```text
(2,2) + Right → (3,2)
```

xác suất transition tương ứng là 1.

---

# 24. Report phải giải thích Bellman

Không được bỏ qua Bellman.

State-value Bellman equation:

$$
V^\pi(s)
=
\sum_a\pi(a|s)
\sum_{s'}P(s'|s,a)
[
R(s,a,s')+\gamma V^\pi(s')
]
$$

Giải thích:

```text
giá trị state hiện tại
=
reward hiện tại
+
giá trị tương lai
```

sau khi xét:

* policy chọn action thế nào
* environment chuyển state thế nào

TD target là one-step bootstrap approximation của ý tưởng Bellman.

---

# 25. Final validation checklist

Trước khi hoàn thành, phải tự kiểm tra:

### Correctness

* [ ] Critic gradient đúng dấu
* [ ] Actor gradient đúng dấu
* [ ] terminal next value = 0
* [ ] softmax numerically stable
* [ ] no NaN
* [ ] no invalid probabilities
* [ ] probabilities sum ≈ 1
* [ ] weights update across episodes
* [ ] Actor/Critic không reset sau mỗi episode

### Environment

* [ ] movement deterministic
* [ ] reward đúng
* [ ] episode termination đúng
* [ ] random training maps
* [ ] unseen test maps

### Evaluation

* [ ] success rate
* [ ] hazard rate
* [ ] timeout rate
* [ ] average return
* [ ] average steps
* [ ] training curve
* [ ] test performance

### Report

* [ ] gọi đúng One-Step Actor–Critic
* [ ] không gọi Q-learning là Critic
* [ ] không nói Q(s,a) = V(s)
* [ ] không nói TD Error = Advantage về mặt định nghĩa
* [ ] giải thích TD Error ≈ Advantage
* [ ] giải thích TD Target
* [ ] giải thích Bellman
* [ ] giải thích state representation
* [ ] giải thích backpropagation
* [ ] phân biệt parameter/hyperparameter
* [ ] report số liệu đúng từ run thực tế

---

# 26. Deliverables

Sau khi hoàn tất, cung cấp:

1. Code environment đã sửa.
2. Actor network đã sửa.
3. Critic network đã sửa.
4. Training notebook/script đã sửa.
5. Evaluation script.
6. Các biểu đồ mới.
7. Một vài sample episodes trên unseen maps.
8. Report được cập nhật hoàn toàn theo code và kết quả thực tế.

**Quan trọng nhất:** không chỉ patch code từng lỗi nhỏ. Hãy kiểm tra lại toàn bộ mathematical consistency giữa:

$$
Loss
\rightarrow
Gradient
\rightarrow
Backpropagation
\rightarrow
Optimizer
\rightarrow
Weight/Bias\ update
$$

và đảm bảo mọi công thức trong report khớp 1-1 với implementation thực tế.
