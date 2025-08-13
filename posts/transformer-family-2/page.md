# Họ Transformer Phiên bản 2.0

Nhiều cải tiến kiến trúc Transformer mới đã được đề xuất kể từ bài đăng cuối cùng của tôi về ["Họ Transformer"](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/) khoảng ba năm trước. Ở đây, tôi đã thực hiện một cuộc tái cấu trúc và bổ sung lớn cho bài viết năm 2020 đó — tái cấu trúc hệ thống các phần và cải thiện nhiều phần bằng các bài báo gần đây hơn. Phiên bản 2.0 là một tập hợp con của phiên bản cũ, dài khoảng gấp đôi.

# Ký hiệu

| Ký hiệu                                                                                                          | Ý nghĩa                                                                                                                                                    |
| :--------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $d$                                                                                                              | Kích thước mô hình / chiều của trạng thái ẩn / kích thước mã hóa vị trí.                                                                                   |
| $h$                                                                                                              | Số lượng đầu (head) trong lớp chú ý đa đầu (multi-head attention).                                                                                         |
| $L$                                                                                                              | Độ dài đoạn của chuỗi đầu vào.                                                                                                                             |
| $N$                                                                                                              | Tổng số lớp chú ý trong mô hình; không xét MoE.                                                                                                            |
| $\mathbf{X} \in \mathbb{R}^{L \times d}$                                                                         | Chuỗi đầu vào trong đó mỗi phần tử đã được ánh xạ thành một vector nhúng (embedding) có dạng $d$, giống như kích thước mô hình.                            |
| $\mathbf{W}^k \in \mathbb{R}^{d \times d_k}$                                                                     | Ma trận trọng số khóa (key).                                                                                                                               |
| $\mathbf{W}^q \in \mathbb{R}^{d \times d_k}$                                                                     | Ma trận trọng số truy vấn (query).                                                                                                                         |
| $\mathbf{W}^v \in \mathbb{R}^{d \times d_v}$                                                                     | Ma trận trọng số giá trị (value). Thường thì $d_k = d_v = d$.                                                                                              |
| $\mathbf{W}^k_i, \mathbf{W}^q_i \in \mathbb{R}^{d \times d_k/h}; \mathbf{W}^v_i \in \mathbb{R}^{d \times d_v/h}$ | Các ma trận trọng số cho mỗi đầu.                                                                                                                          |
| $\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$                                                                     | Ma trận trọng số đầu ra.                                                                                                                                   |
| $\mathbf{Q} = \mathbf{X}\mathbf{W}^q \in \mathbb{R}^{L \times d_k}$                                              | Các đầu vào nhúng của truy vấn.                                                                                                                            |
| $\mathbf{K} = \mathbf{X}\mathbf{W}^k \in \mathbb{R}^{L \times d_k}$                                              | Các đầu vào nhúng của khóa.                                                                                                                                |
| $\mathbf{V} = \mathbf{X}\mathbf{W}^v \in \mathbb{R}^{L \times d_v}$                                              | Các đầu vào nhúng của giá trị.                                                                                                                             |
| $\mathbf{q}_i, \mathbf{k}_i \in \mathbb{R}^{d_k}, \mathbf{v}_i \in \mathbb{R}^{d_v}$                             | Các vector hàng trong ma trận truy vấn, khóa và giá trị, $\mathbf{Q}$, $\mathbf{K}$ và $\mathbf{V}$.                                                       |
| $S_i$                                                                                                            | Một tập hợp các vị trí khóa để truy vấn thứ $i$, $\mathbf{q}_i$, chú ý đến.                                                                                |
| $\mathbf{A} \in \mathbb{R}^{L \times L}$                                                                         | Ma trận tự chú ý (self-attention) giữa một chuỗi đầu vào có độ dài $L$ và chính nó. $\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})$. |
| $a_{ij} \in \mathbf{A}$                                                                                          | Điểm chú ý vô hướng giữa truy vấn $\mathbf{q}_i$ và khóa $\mathbf{k}_j$.                                                                                   |
| $\mathbf{P} \in \mathbb{R}^{L \times d}$                                                                         | Ma trận mã hóa vị trí, trong đó hàng thứ $i$, $\mathbf{p}_i$, là mã hóa vị trí cho đầu vào $\mathbf{x}_i$.                                                 |

# Kiến thức cơ bản về Transformer

Mô hình **Transformer** (sẽ được gọi là "vanilla Transformer" để phân biệt với các phiên bản nâng cao khác; [Vaswani, et al., 2017](https://arxiv.org/abs/1706.03762)) có kiến trúc mã hóa-giải mã (encoder-decoder), thường được sử dụng trong nhiều mô hình [NMT](https://lilianweng.github.io/posts/2018-06-24-attention/#born-for-translation). Sau này, Transformer đơn giản hóa đã được chứng minh là đạt hiệu suất tuyệt vời trong các tác vụ mô hình hóa ngôn ngữ, như trong [BERT](https://lilianweng.github.io/posts/2019-01-31-lm/#bert) chỉ có bộ mã hóa hoặc [GPT](https://lilianweng.github.io/posts/2019-01-31-lm/#openai-gpt) chỉ có bộ giải mã.

## Cơ chế chú ý và Tự chú ý (Attention and Self-Attention)

**Chú ý (Attention)** là một cơ chế trong mạng nơ-ron mà một mô hình có thể học cách đưa ra dự đoán bằng cách chú ý có chọn lọc đến một tập dữ liệu nhất định. Lượng chú ý được định lượng bằng các trọng số đã học và do đó đầu ra thường được hình thành dưới dạng trung bình có trọng số.

**Tự chú ý (Self-attention)** là một loại cơ chế chú ý trong đó mô hình đưa ra dự đoán cho một phần của mẫu dữ liệu bằng cách sử dụng các phần khác của quan sát về cùng một mẫu. Về mặt khái niệm, nó khá giống với [phương pháp trung bình phi cục bộ (non-local means)](https://en.wikipedia.org/wiki/Non-local_means). Cũng cần lưu ý rằng tự chú ý là bất biến hoán vị; nói cách khác, nó là một hoạt động trên các tập hợp.

Có nhiều dạng chú ý / tự chú ý khác nhau, Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) dựa trên _chú ý tích vô hướng có trọng số (scaled dot-product attention)_: cho một ma trận truy vấn (query) $\mathbf{Q}$, một ma trận khóa (key) $\mathbf{K}$ và một ma trận giá trị (value) $\mathbf{V}$, đầu ra là một tổng có trọng số của các vector giá trị, trong đó trọng số được gán cho mỗi vị trí giá trị được xác định bởi tích vô hướng của truy vấn với khóa tương ứng:

$$
\text{attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\frac{\mathbf{Q} {\mathbf{K}}^\top}{\sqrt{d_k}})\mathbf{V}
$$

Và đối với một vector truy vấn và một vector khóa $\mathbf{q}_i, \mathbf{k}_j \in \mathbb{R}^d$ (các vector hàng trong ma trận truy vấn và khóa), chúng ta có một điểm số vô hướng:

$$
a_{ij} = \text{softmax}(\frac{\mathbf{q}_i {\mathbf{k}_j}^\top}{\sqrt{d_k}})
= \frac{\exp(\frac{\mathbf{q}_i {\mathbf{k}_j}^\top}{\sqrt{d_k}})}{ \sum_{r \in \mathcal{S}_i} \exp(\frac{\mathbf{q}_i {\mathbf{k}_r}^\top}{\sqrt{d_k}}) }
$$

trong đó $\mathcal{S}_i$ là một tập hợp các vị trí khóa mà truy vấn thứ $i$ cần chú ý đến.

Xem [bài viết cũ của tôi để biết các loại chú ý khác](https://lilianweng.github.io/posts/2018-06-24-attention/#a-family-of-attention-mechanisms) nếu bạn quan tâm.

## Tự chú ý đa đầu (Multi-Head Self-Attention)

Mô-đun **tự chú ý đa đầu (multi-head self-attention)** là một thành phần quan trọng trong Transformer. Thay vì chỉ tính toán sự chú ý một lần, cơ chế đa đầu chia nhỏ các đầu vào thành các phần nhỏ hơn và sau đó tính toán sự chú ý tích vô hướng có trọng số trên mỗi không gian con một cách song song. Các đầu ra chú ý độc lập chỉ đơn giản là được nối lại và biến đổi tuyến tính thành các kích thước mong đợi.

$$
\begin{aligned}
\text{MultiHeadAttn}(\mathbf{X}_q, \mathbf{X}_k, \mathbf{X}_v) &= [\text{head}_1; \dots; \text{head}_h] \mathbf{W}^o \\
\text{where head}_i &= \text{Attention}(\mathbf{X}_q\mathbf{W}^q_i, \mathbf{X}_k\mathbf{W}^k_i, \mathbf{X}_v\mathbf{W}^v_i)
\end{aligned}
$$

trong đó $[.;.]$ là một phép toán nối (concatenation). $\mathbf{W}^q_i, \mathbf{W}^k_i \in \mathbb{R}^{d \times d_k/h}, \mathbf{W}^v_i \in \mathbb{R}^{d \times d_v/h}$ là các ma trận trọng số để ánh xạ các embedding đầu vào có kích thước $L \times d$ thành các ma trận truy vấn, khóa và giá trị. Và $\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$ là biến đổi tuyến tính đầu ra. Tất cả các trọng số đều phải được học trong quá trình huấn luyện.

![Minh họa về cơ chế chú ý tích vô hướng có trọng số đa đầu.](/posts/transformer-family-2/multi-head-attention.png)
_Minh họa về cơ chế chú ý tích vô hướng có trọng số đa đầu. (Nguồn ảnh: Hình 2 trong [Vaswani, et al., 2017](https://arxiv.org/abs/1706.03762))_

## Kiến trúc Mã hóa-Giải mã (Encoder-Decoder Architecture)

**Bộ mã hóa (encoder)** tạo ra một biểu diễn dựa trên sự chú ý có khả năng xác định một mẩu thông tin cụ thể từ một ngữ cảnh lớn. Nó bao gồm một chồng 6 mô-đun giống hệt nhau, mỗi mô-đun chứa hai mô-đun con, một lớp _tự chú ý đa đầu_ và một mạng truyền thẳng kết nối đầy đủ _theo từng điểm (point-wise)_. Theo từng điểm có nghĩa là nó áp dụng cùng một phép biến đổi tuyến tính (với cùng trọng số) cho mỗi phần tử trong chuỗi. Điều này cũng có thể được xem như một lớp tích chập với kích thước bộ lọc là 1. Mỗi mô-đun con có một kết nối phần dư (residual connection) và chuẩn hóa lớp (layer normalization). Tất cả các mô-đun con đều cho ra dữ liệu có cùng chiều $d$.

Chức năng của **bộ giải mã (decoder)** của Transformer là truy xuất thông tin từ biểu diễn đã được mã hóa. Kiến trúc này khá giống với bộ mã hóa, ngoại trừ việc bộ giải mã chứa hai mô-đun con chú ý đa đầu thay vì một trong mỗi mô-đun lặp lại giống hệt nhau. Mô-đun con chú ý đa đầu đầu tiên được _che (masked)_ để ngăn các vị trí chú ý đến tương lai.

![Kiến trúc của mô hình vanilla Transformer.](/posts/transformer-family-2/transformer.png)
_Kiến trúc của mô hình vanilla Transformer. (Nguồn ảnh: [Hình 17](https://lilianweng.github.io/posts/2018-06-24-attention/#full-architecture))_

## Mã hóa vị trí (Positional Encoding)

Bởi vì phép toán tự chú ý là bất biến hoán vị, điều quan trọng là phải sử dụng **mã hóa vị trí** phù hợp để cung cấp _thông tin về thứ tự_ cho mô hình. Mã hóa vị trí $\mathbf{P} \in \mathbb{R}^{L \times d}$ có cùng chiều với embedding đầu vào, vì vậy nó có thể được cộng trực tiếp vào đầu vào. Vanilla Transformer đã xem xét hai loại mã hóa:

### Mã hóa vị trí hình sin

Mã hóa vị trí hình sin được định nghĩa như sau, với vị trí của token là $i=1,\dots,L$ và chiều là $\delta=1,\dots,d$:

$$
\text{PE}(i,\delta) =
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
\cos(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
\end{cases}
$$

Bằng cách này, mỗi chiều của mã hóa vị trí tương ứng với một hình sin có bước sóng khác nhau ở các chiều khác nhau, từ $2\pi$ đến $10000 \cdot 2\pi$.

![Mã hóa vị trí hình sin với L=32 và d=128.](/posts/transformer-family-2/sinoidual-positional-encoding.png)
_Mã hóa vị trí hình sin với $L=32$ và $d=128$. Giá trị nằm trong khoảng từ -1 (đen) đến 1 (trắng) và giá trị 0 có màu xám._

### Mã hóa vị trí đã học

Mã hóa vị trí đã học gán cho mỗi phần tử một vector cột _đã học_ để mã hóa vị trí tuyệt đối của nó ([Gehring, et al. 2017](https://arxiv.org/abs/1705.03122)) và hơn nữa, mã hóa này có thể được học khác nhau cho mỗi lớp ([Al-Rfou et al. 2018](https://arxiv.org/abs/1808.04444)).

### Mã hóa vị trí tương đối

[Shaw et al. (2018)](https://arxiv.org/abs/1803.02155) đã kết hợp thông tin vị trí tương đối vào $\mathbf{W}^k$ và $\mathbf{W}^v$. Vị trí tương đối tối đa được cắt ở một giá trị tuyệt đối tối đa là $k$ và hoạt động cắt này cho phép mô hình khái quát hóa cho các độ dài chuỗi chưa từng thấy. Do đó, có $2k + 1$ nhãn cạnh duy nhất được xem xét và chúng ta ký hiệu $\mathbf{P}^k, \mathbf{P}^v \in \mathbb{R}^{2k+1}$ là các biểu diễn vị trí tương đối có thể học được.

$$
A_{ij}^k = P^k_{\text{clip}(j - i, k)} \quad
A_{ij}^v = P^v_{\text{clip}(j - i, k)} \quad
\text{where }\text{clip}(x, k) = \text{clip}(x, -k, k)
$$

<a id="transformer-xl-encoding"></a>[Transformer-XL](#transformer-xl) ([Dai et al., 2019](https://arxiv.org/abs/1901.02860)) đã đề xuất một loại mã hóa vị trí tương đối dựa trên việc tham số hóa lại tích vô hướng của các khóa và truy vấn. Để giữ cho luồng thông tin vị trí được kết nối mạch lạc giữa các đoạn, Transformer-XL thay vào đó mã hóa vị trí _tương đối_, vì chỉ cần biết độ lệch vị trí để đưa ra dự đoán tốt, tức là $i-j$, giữa một vector khóa $\mathbf{k}_{\tau, j}$ và truy vấn của nó $\mathbf{q}_{\tau, i}$.

Nếu bỏ qua hệ số vô hướng $1/\sqrt{d_k}$ và thành phần chuẩn hóa trong softmax nhưng bao gồm mã hóa vị trí, chúng ta có thể viết điểm chú ý giữa truy vấn tại vị trí $i$ và khóa tại vị trí $j$ như sau:

$$
\begin{aligned}
a_{ij}
&= \mathbf{q}_i {\mathbf{k}_j}^\top = (\mathbf{x}_i + \mathbf{p}_i)\mathbf{W}^q ((\mathbf{x}_j + \mathbf{p}_j)\mathbf{W}^k)^\top \\
&= \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{x}_j^\top + \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{p}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{x}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top\mathbf{p}_j^\top
\end{aligned}
$$

Transformer-XL tái tham số hóa bốn thành phần trên như sau:

$$
a_{ij}^\text{rel} =
\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_\text{định vị dựa trên nội dung} +
\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_\text{thiên vị vị trí phụ thuộc vào nội dung} +
\underbrace{ \color{red}{\mathbf{u}} \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_\text{thiên vị nội dung toàn cục} +
\underbrace{ \color{red}{\mathbf{v}} \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_\text{thiên vị vị trí toàn cục}
$$

- Thay thế $\mathbf{p}_j$ bằng mã hóa vị trí tương đối $\mathbf{r}_{i-j} \in \mathbf{R}^{d}$;
- Thay thế $\mathbf{p}_i\mathbf{W}^q$ bằng hai tham số có thể huấn luyện là $\mathbf{u}$ (cho nội dung) và $\mathbf{v}$ (cho vị trí) trong hai thuật ngữ khác nhau;
- Chia $\mathbf{W}^k$ thành hai ma trận, $\mathbf{W}^k_E$ cho thông tin nội dung và $\mathbf{W}^k_R$ cho thông tin vị trí.

### Nhúng vị trí quay (Rotary Position Embedding)

Nhúng vị trí quay (_RoPE_; [Su et al. 2021](https://arxiv.org/abs/2104.09864)) mã hóa vị trí tuyệt đối bằng một [ma trận quay](https://en.wikipedia.org/wiki/Rotation_matrix) và nhân các ma trận khóa và giá trị của mọi lớp chú ý với nó để đưa thông tin vị trí tương đối vào mọi lớp.

Khi mã hóa thông tin vị trí tương đối vào tích vô hướng của khóa thứ $i$ và truy vấn thứ $j$, chúng ta muốn xây dựng hàm sao cho tích vô hướng chỉ liên quan đến vị trí tương đối $i-j$. Nhúng Vị trí Quay (RoPE) tận dụng phép toán quay trong không gian Euclide và coi nhúng vị trí tương đối đơn giản là quay ma trận đặc trưng một góc tỷ lệ với chỉ số vị trí của nó.

Cho một vector $\mathbf{z}$, nếu chúng ta muốn quay nó ngược chiều kim đồng hồ một góc $\theta$, chúng ta có thể nhân nó với một ma trận quay để được $R\mathbf{z}$, trong đó ma trận quay $R$ được định nghĩa là:

$$
R = \begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

Khi tổng quát hóa cho không gian có chiều cao hơn, RoPE chia không gian $d$-chiều thành $d/2$ không gian con và xây dựng một ma trận quay $R$ có kích thước $d \times d$ cho token tại vị trí $i$:

$$
R^d_{\Theta, i} = \begin{bmatrix}
\cos i\theta_1 & -\sin i\theta_1 & 0 & 0 & \dots & 0 & 0 \\
\sin i\theta_1 & \cos i\theta_1 & 0 & 0 & \dots & 0 & 0 \\
0 & 0 & \cos i\theta_2 & -\sin i\theta_2 & \dots & 0 & 0 \\
0 & 0 & \sin i\theta_2 & \cos i\theta_2 & \dots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \dots & \cos i\theta_{d/2} & -\sin i\theta_{d/2} \\
0 & 0 & 0 & 0 & \dots & \sin i\theta_{d/2} & \cos i\theta_{d/2} \\
\end{bmatrix}
$$

trong bài báo, chúng ta có $\Theta = \{\theta_i = 10000^{-2(i−1)/d}, i \in [1, 2, \dots, d/2]\}$. Lưu ý rằng điều này về cơ bản tương đương với mã hóa vị trí hình sin nhưng được xây dựng dưới dạng ma trận quay.

Sau đó, cả ma trận khóa và ma trận truy vấn đều kết hợp thông tin vị trí bằng cách nhân với ma trận quay này:

$$
\begin{aligned}
& \mathbf{q}_i^\top \mathbf{k}_j = (R^d_{\Theta, i} \mathbf{W}^q\mathbf{x}_i)^\top (R^d_{\Theta, j} \mathbf{W}^k\mathbf{x}_j) = \mathbf{x}_i^\top\mathbf{W}^q R^d_{\Theta, j-i}\mathbf{W}^k\mathbf{x}_j \\
& \text{ where } R^d_{\Theta, j-i} = (R^d_{\Theta, i})^\top R^d_{\Theta, j}
\end{aligned}
$$

![Minh họa trực quan về cách triển khai nhúng vị trí quay.](/posts/transformer-family-2/RoPE.png)
_Minh họa trực quan về cách triển khai nhúng vị trí quay. (Nguồn ảnh: [Su et al., 2021](https://arxiv.org/abs/2104.09864)) Lưu ý: Tôi đã sử dụng $i$ thay vì $m$ để biểu thị chỉ số vị trí so với hình gốc trong bài báo._

# Ngữ cảnh dài hơn

Độ dài của một chuỗi đầu vào cho các mô hình transformer tại thời điểm suy luận bị giới hạn trên bởi độ dài ngữ cảnh được sử dụng để huấn luyện. Việc tăng độ dài ngữ cảnh một cách ngây thơ dẫn đến tiêu thụ nhiều cả về thời gian ($\mathcal{O}(L^2d)$) và bộ nhớ ($\mathcal{O}(L^2)$) và có thể không được hỗ trợ do hạn chế về phần cứng.

Phần này giới thiệu một số cải tiến trong kiến trúc transformer để hỗ trợ tốt hơn cho ngữ cảnh dài tại thời điểm suy luận; ví dụ: sử dụng bộ nhớ bổ sung, thiết kế để ngoại suy ngữ cảnh tốt hơn hoặc cơ chế lặp lại.

## Bộ nhớ Ngữ cảnh

Vanilla Transformer có một khoảng chú ý cố định và hạn chế. Mô hình chỉ có thể chú ý đến các phần tử khác trong cùng một đoạn trong mỗi bước cập nhật và không có thông tin nào có thể chảy qua các đoạn có độ dài cố định riêng biệt. _Sự phân đoạn ngữ cảnh_ này gây ra một số vấn đề:

- Mô hình không thể nắm bắt được các phụ thuộc dài hạn.
- Rất khó để dự đoán một vài token đầu tiên trong mỗi đoạn khi không có hoặc có ít ngữ cảnh.
- Việc đánh giá rất tốn kém. Bất cứ khi nào đoạn được dịch sang phải một vị trí, đoạn mới sẽ được xử lý lại từ đầu, mặc dù có rất nhiều token chồng chéo.

<a id="transformer-xl"></a>**Transformer-XL** ([Dai et al., 2019](https://arxiv.org/abs/1901.02860); "XL" có nghĩa là "extra long") sửa đổi kiến trúc để tái sử dụng các trạng thái ẩn giữa các đoạn với một bộ nhớ bổ sung. Kết nối lặp lại giữa các đoạn được đưa vào mô hình bằng cách liên tục sử dụng các trạng thái ẩn từ các đoạn trước đó.

![So sánh giữa giai đoạn huấn luyện của vanilla Transformer và Transformer-XL với độ dài đoạn là 4.](/posts/transformer-family-2/transformer-XL-training.png)
_So sánh giữa giai đoạn huấn luyện của vanilla Transformer và Transformer-XL với độ dài đoạn là 4. (Nguồn ảnh: phần bên trái của Hình 2 trong [Dai et al., 2019](https://arxiv.org/abs/1901.02860))._

Hãy ký hiệu trạng thái ẩn của lớp thứ $n$ cho đoạn thứ $(\tau + 1)$ trong mô hình là $\mathbf{h}_{\tau+1}^{(n)} \in \mathbb{R}^{L \times d}$. Ngoài trạng thái ẩn của lớp cuối cùng cho cùng một đoạn $\mathbf{h}_{\tau+1}^{(n-1)}$, nó còn phụ thuộc vào trạng thái ẩn của cùng một lớp cho đoạn trước đó $\mathbf{h}_{\tau}^{(n)}$. Bằng cách kết hợp thông tin từ các trạng thái ẩn trước đó, mô hình mở rộng phạm vi chú ý của mình xa hơn nhiều trong quá khứ, qua nhiều đoạn.

$$
\begin{aligned}
\color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} &= [\text{stop-gradient}(\mathbf{h}_{\tau}^{(n-1)}) \circ \mathbf{h}_{\tau+1}^{(n-1)}] \\
\mathbf{Q}_{\tau+1}^{(n)} &= \mathbf{h}_{\tau+1}^{(n-1)}\mathbf{W}^q \\
\mathbf{K}_{\tau+1}^{(n)} &= \color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} \mathbf{W}^k \\
\mathbf{V}_{\tau+1}^{(n)} &= \color{red}{\widetilde{\mathbf{h}}_{\tau+1}^{(n-1)}} \mathbf{W}^v \\
\mathbf{h}_{\tau+1}^{(n)} &= \text{transformer-layer}(\mathbf{Q}_{\tau+1}^{(n)}, \mathbf{K}_{\tau+1}^{(n)}, \mathbf{V}_{\tau+1}^{(n)})
\end{aligned}
$$

Lưu ý rằng cả khóa và giá trị đều dựa vào các trạng thái ẩn mở rộng, trong khi truy vấn chỉ sử dụng các trạng thái ẩn ở bước hiện tại. Phép toán nối $[. \circ .]$ được thực hiện theo chiều dài chuỗi. Và Transformer-XL cần sử dụng [mã hóa vị trí tương đối](#transformer-xl-encoding) vì các đoạn trước và hiện tại sẽ được gán cùng một mã hóa nếu chúng ta mã hóa các vị trí tuyệt đối, điều này là không mong muốn.

**Transformer nén (Compressive Transformer)** ([Rae et al. 2019](https://arxiv.org/abs/1911.05507)) mở rộng Transformer-XL bằng cách nén các bộ nhớ trong quá khứ để hỗ trợ các chuỗi dài hơn. Nó thêm rõ ràng _các khe cắm bộ nhớ_ có kích thước $m_m$ cho mỗi lớp để lưu trữ các kích hoạt trong quá khứ của lớp này nhằm bảo toàn ngữ cảnh dài. Khi một số kích hoạt trong quá khứ đủ cũ, chúng sẽ được nén và lưu trong một _bộ nhớ nén_ bổ sung có kích thước $m_{cm}$ cho mỗi lớp.

![Transformer nén duy trì hai loại khe cắm bộ nhớ, bộ nhớ và bộ nhớ nén, để hỗ trợ ngữ cảnh dài.](/posts/transformer-family-2/compressive-transformer.png)
_Transformer nén duy trì hai loại khe cắm bộ nhớ, bộ nhớ và bộ nhớ nén, để hỗ trợ ngữ cảnh dài. (Nguồn ảnh: [Rae et al. 2019](https://arxiv.org/abs/1911.05507))._

Cả bộ nhớ và bộ nhớ nén đều là hàng đợi FIFO. Cho độ dài ngữ cảnh của mô hình là $L$, hàm nén với tỷ lệ nén $c$ được định nghĩa là $f_c: \mathbb{R}^{L \times d} \to \mathbb{R}^{[\frac{L}{c}] \times d}$, ánh xạ $L$ kích hoạt cũ nhất thành $[\frac{L}{c}]$ phần tử bộ nhớ nén. Có một số lựa chọn về hàm nén:

1.  Gộp max/trung bình với kernel và bước có kích thước $c$;
2.  Tích chập 1D với kernel và bước có kích thước $c$ (cần học thêm các tham số);
3.  Tích chập giãn nở (cần học thêm các tham số). Trong các thí nghiệm của họ, nén tích chập hoạt động tốt nhất trên bộ dữ liệu `EnWik8`;
4.  Các bộ nhớ được sử dụng nhiều nhất.

Transformer nén có hai hàm mất mát huấn luyện bổ sung:

1.  **Mất mát tự mã hóa** (mục tiêu nén không mất dữ liệu) đo lường mức độ chúng ta có thể tái tạo lại các bộ nhớ ban đầu từ các bộ nhớ nén.

    $$
    \mathcal{L}_{ac} = \left\| \mathbf{old\_mem}^{(i)} - g\big(\mathbf{new\_cm}^{(i)}\big) \right\|_2
    $$

    trong đó $g: \mathbb{R}^{[\frac{L}{c}] \times d} \to \mathbb{R}^{L \times d}$ đảo ngược hàm nén $f$.

2.  **Mất mát tái tạo chú ý** (mục tiêu có mất mát) tái tạo lại sự chú ý dựa trên nội dung trên bộ nhớ so với bộ nhớ nén và giảm thiểu sự khác biệt:
    $$
    \mathcal{L}_{ar} = \left\| \operatorname{attn}\big(\mathbf{h}^{(i)}, \mathbf{old\_mem}^{(i)}\big) - \operatorname{attn}\big(\mathbf{h}^{(i)}, \mathbf{new\_cm}^{(i)}\big) \right\|_2
    $$

Transformer-XL với bộ nhớ có kích thước $m$ có phạm vi thời gian tối đa là $m \times N$, trong đó $N$ là số lớp trong mô hình và chi phí chú ý là $\mathcal{O}(L^2 + Lm)$. So sánh, transformer nén có phạm vi thời gian là $(m_m + c \cdot m_{cm}) \times N$ và chi phí chú ý là $\mathcal{O}(L^2 + L(m_m + m_{cm}))$. Tỷ lệ nén $c$ lớn hơn mang lại sự cân bằng tốt hơn giữa độ dài phạm vi thời gian và chi phí chú ý.

Trọng số chú ý, từ cũ nhất đến mới nhất, được lưu trữ ở ba vị trí: bộ nhớ nén → bộ nhớ → chuỗi được che theo quan hệ nhân quả. Trong các thí nghiệm, họ đã quan sát thấy sự gia tăng trọng số chú ý từ các kích hoạt cũ nhất được lưu trữ trong bộ nhớ thông thường đến các kích hoạt được lưu trữ trong bộ nhớ nén, điều này ngụ ý rằng mạng đang học cách bảo tồn thông tin nổi bật.

![Trọng số chú ý với một độ lệch chuẩn làm thanh lỗi so với các vị trí bộ nhớ, từ cũ nhất (trái) đến mới nhất (phải).](/posts/transformer-family-2/compressive-transformer-memory.png)
_Trọng số chú ý với một độ lệch chuẩn làm thanh lỗi so với các vị trí bộ nhớ, từ cũ nhất (trái) đến mới nhất (phải). (Nguồn ảnh: [Rae et al. 2019](https://arxiv.org/abs/1911.05507))._

## Bộ nhớ ngoài không khả vi

**$k$NN-LM** ([Khandelwal et al. 2020](https://arxiv.org/abs/1911.00172)) tăng cường một LM được huấn luyện trước bằng một mô hình $k$NN riêng biệt bằng cách nội suy tuyến tính các xác suất của token tiếp theo được dự đoán bởi cả hai mô hình. Mô hình $k$NN được xây dựng trên một kho lưu trữ khóa-giá trị bên ngoài có thể lưu trữ bất kỳ tập dữ liệu huấn luyện trước lớn nào hoặc tập dữ liệu OOD mới. Kho dữ liệu này được tiền xử lý để lưu một số lượng _lớn_ các cặp, (biểu diễn nhúng LM của ngữ cảnh, token tiếp theo) và việc truy xuất hàng xóm gần nhất xảy ra trong không gian nhúng của LM. Bởi vì kho dữ liệu có thể rất lớn, chúng ta cần dựa vào các thư viện để tìm kiếm vector dày đặc nhanh chóng như [FAISS](https://github.com/facebookresearch/faiss) hoặc [ScaNN](https://github.com/google-research/google-research/tree/master/scann). Quá trình lập chỉ mục chỉ xảy ra một lần và tính song song dễ dàng thực hiện tại thời điểm suy luận.

Tại thời điểm suy luận, xác suất của token tiếp theo là tổng có trọng số của hai dự đoán:

$$
\begin{aligned}
p(y \vert \mathbf{x}) &= \lambda \; p_\text{kNN}(y \vert \mathbf{x}) + (1- \lambda) \; p_\text{LM}(y \vert \mathbf{x}) \\
p_\text{kNN}(y \vert \mathbf{x}) &\propto \sum_{(k_i, w_i) \in \mathcal{N}} \mathbb{1}[y = w_i] \exp(-d(k_i, f(\mathbf{x})))
\end{aligned}
$$

trong đó $\mathcal{N}$ chứa một tập hợp các điểm dữ liệu hàng xóm gần nhất được truy xuất bởi $k$NN; $d(., .)$ là một hàm khoảng cách như khoảng cách L2.

Theo các thí nghiệm, kích thước kho dữ liệu lớn hơn hoặc $k$ lớn hơn có tương quan với perplexity tốt hơn. Vô hướng trọng số $\lambda$ nên được điều chỉnh, nhưng nói chung, dự kiến sẽ lớn hơn đối với dữ liệu ngoài miền so với dữ liệu trong miền và một kho dữ liệu lớn hơn có thể có $\lambda$ lớn hơn.

**SPALM** (_mô hình ngôn ngữ bán tham số thích ứng_; [Yogatama et al. 2021](https://arxiv.org/abs/2102.02557)) kết hợp cả (1) bộ nhớ kiểu Transformer-XL cho các trạng thái ẩn từ ngữ cảnh bên ngoài làm bộ nhớ ngắn hạn và (2) kho lưu trữ khóa-giá trị kiểu $k$NN-LM làm bộ nhớ dài hạn.

![Minh họa về cách SPALM kết hợp bộ nhớ ngữ cảnh của các trạng thái ẩn trong quá khứ (bộ nhớ ngắn hạn) với một kho dữ liệu khóa-giá trị bên ngoài (bộ nhớ dài hạn) để hỗ trợ ngữ cảnh dài hơn.](/posts/transformer-family-2/SPALM2.png)
_Minh họa về cách SPALM kết hợp bộ nhớ ngữ cảnh của các trạng thái ẩn trong quá khứ (bộ nhớ ngắn hạn) với một kho dữ liệu khóa-giá trị bên ngoài (bộ nhớ dài hạn) để hỗ trợ ngữ cảnh dài hơn. (Nguồn ảnh: [Yogatama et al. 2021](https://arxiv.org/abs/2102.02557))._

SPALM chạy tìm kiếm $k$NN để lấy $k$ token có ngữ cảnh liên quan nhất. Đối với mỗi token, chúng ta có thể nhận được cùng một biểu diễn nhúng được cung cấp bởi một LM được huấn luyện trước, được ký hiệu là $\{\mathbf{y}_i\}_{i=1}^k$. Cơ chế cổng (gating) đầu tiên tổng hợp các nhúng token được truy xuất bằng một lớp chú ý đơn giản sử dụng $\mathbf{h}^R_t$ (trạng thái ẩn cho token $x_t$ ở lớp $R$) làm truy vấn và sau đó học một tham số cổng $\mathbf{g}_t$ để cân bằng giữa thông tin cục bộ $\mathbf{h}^R_t$ và thông tin dài hạn $\mathbf{m}_t$.

$$
\begin{aligned}
\mathbf{m}_t &= \sum_{i=1}^k \frac{\exp(\mathbf{y}_i^\top \mathbf{h}^R_t)}{\sum_{j=1}^k \exp(\mathbf{y}_j^\top \mathbf{h}^R_t)} \cdot \mathbf{y}_i \\
\mathbf{g}_t &= \sigma(\mathbf{w}_g^\top \mathbf{h}_t^R) \\
\mathbf{z}_t &= (1 - \mathbf{g}_t) \odot \mathbf{m}_t + \mathbf{g}_t \odot \mathbf{h}^R_t \\
p(x_{t+1}\mid \mathbf{x}_{\leq t}) &= \text{softmax}(\mathbf{z}_t; \mathbf{W})
\end{aligned}
$$

trong đó $\mathbf{w}_g$ là một vector tham số cần học; $\sigma(.)$ là hàm sigmoid; $\mathbf{W}$ là ma trận nhúng từ được chia sẻ giữa cả token đầu vào và đầu ra. Khác với $k$NN-LM, họ không thấy khoảng cách đến hàng xóm gần nhất hữu ích trong việc tổng hợp các token được truy xuất.

Trong quá trình huấn luyện, các biểu diễn khóa trong bộ nhớ dài hạn không đổi, được tạo ra bởi một LM được huấn luyện trước, nhưng bộ mã hóa giá trị, tức là ma trận nhúng từ, được cập nhật.

**Transformer ghi nhớ (Memorizing Transformer)** ([Wu et al. 2022](https://arxiv.org/abs/2203.08913)) thêm một lớp chú ý tăng cường $k$NN gần đỉnh của một Transformer chỉ có bộ giải mã. Lớp đặc biệt này duy trì một bộ đệm FIFO kiểu Transformer-XL của các cặp khóa-giá trị trong quá khứ.

Các giá trị QKV tương tự được sử dụng cho cả cơ chế chú ý cục bộ và $k$NN. Tra cứu $k$NN trả về $k$ cặp (khóa, giá trị) hàng đầu cho mỗi truy vấn trong chuỗi đầu vào và sau đó chúng được xử lý thông qua chồng tự chú ý để tính toán giá trị trung bình có trọng số của các giá trị được truy xuất. Hai loại chú ý được kết hợp với một tham số cổng có thể học được cho mỗi đầu. Để ngăn chặn sự thay đổi lớn về phân phối trong độ lớn của giá trị, cả khóa và giá trị trong bộ đệm đều được chuẩn hóa.

Những gì họ tìm thấy trong các thí nghiệm với Transformer Ghi nhớ:

- Trong một số thí nghiệm, người ta quan sát thấy rằng việc huấn luyện các mô hình có bộ nhớ nhỏ và sau đó tinh chỉnh với bộ nhớ lớn hơn sẽ hoạt động tốt hơn so với việc huấn luyện với bộ nhớ lớn ngay từ đầu.
- Transformer Ghi nhớ nhỏ hơn chỉ với 8k token trong bộ nhớ có thể đạt được độ phức tạp tương đương với một Transformer vanilla lớn hơn có số lượng tham số có thể huấn luyện gấp 5 lần.
- Việc tăng kích thước của bộ nhớ ngoài đã mang lại lợi ích nhất quán lên đến kích thước 262K.
- Một transformer không có bộ nhớ có thể được tinh chỉnh để sử dụng bộ nhớ.

![Việc tinh chỉnh một Transformer vanilla với bộ nhớ khóa-giá trị có thể đạt được hiệu suất tương tự như việc huấn luyện một transformer ghi nhớ từ đầu.](/posts/transformer-family-2/memorizing-transformer.png)
_Việc tinh chỉnh một Transformer vanilla với bộ nhớ khóa-giá trị có thể đạt được hiệu suất tương tự như việc huấn luyện một transformer ghi nhớ từ đầu. (Nguồn ảnh: [Wu et al. 2022](https://arxiv.org/abs/2203.08913))._

## Điểm Chú ý được Tăng cường bằng Khoảng cách

**Transformer Nhận biết Khoảng cách (DA-Transformer)** ([Wu, et al. 2021](https://aclanthology.org/2021.naacl-main.166)) và **Chú ý với Thiên vị Tuyến tính (ALiBi)** ([Press et al. 2022](https://arxiv.org/abs/2108.12409)) được thúc đẩy bởi các ý tưởng tương tự — để khuyến khích mô hình ngoại suy trên ngữ cảnh dài hơn so với những gì mô hình được huấn luyện, chúng ta có thể gắn thông tin vị trí vào mọi cặp điểm chú ý một cách rõ ràng dựa trên khoảng cách giữa các token khóa và truy vấn.

Lưu ý rằng mã hóa vị trí mặc định trong vanilla Transformer chỉ thêm thông tin vị trí vào chuỗi đầu vào, trong khi các cơ chế mã hóa được cải tiến sau này thay đổi điểm chú ý của mọi lớp, chẳng hạn như [nhúng vị trí quay](#rotary-position-embedding), và chúng có dạng rất giống với các điểm chú ý được tăng cường bằng khoảng cách.

_DA-Transformer_ ([Wu, et al. 2021](https://aclanthology.org/2021.naacl-main.166)) nhân các điểm chú ý ở mỗi lớp với một độ lệch có thể học được được xây dựng dưới dạng một hàm của khoảng cách giữa khóa và truy vấn. Các đầu chú ý khác nhau sử dụng các tham số khác nhau để phân biệt các sở thích đa dạng đối với ngữ cảnh ngắn hạn và dài hạn. Cho hai vị trí, $i, j$, DA-Transformer sử dụng hàm trọng số sau để thay đổi điểm tự chú ý:

$$
\begin{aligned}
\mathbf{R}^{(i)} &= \alpha_i \mathbf{R} \quad \text{where }R_{ij} = \vert i-j \vert\\
f(\mathbf{R}^{(i)}; \beta_i) &= \frac{1 + \exp(\beta_i)}{1 + \exp(\beta_i - \mathbf{R}^{(i)})} \\
\text{attn}(\mathbf{Q}^{(i)}, \mathbf{K}^{(i)}, \mathbf{V}^{(i)}) &= \text{row-softmax}\Big(\frac{\text{ReLU}(\mathbf{Q}^{(i)}\mathbf{K}^{(i)\top})f(\mathbf{R}^{(i)})}{\sqrt{d}}\Big) \mathbf{V}^{(i)}
\end{aligned}
$$

trong đó $\alpha_i$ là một tham số có thể học được để trọng số khoảng cách tương đối khác nhau cho mỗi đầu, trong đó đầu được lập chỉ mục bằng chỉ số trên $^{(i)}$; $\beta_i$ là một tham số có thể học được để kiểm soát giới hạn trên và độ dốc tăng dần theo khoảng cách cho đầu chú ý thứ $i$. Hàm trọng số $f(.)$ được thiết kế sao cho: (1) $f(0)=1$; (2) $f(\mathbf{R}^{(i)}) = 0$ khi $\mathbf{R}^{(i)} \to -\infty$; (3) $f(\mathbf{R}^{(i)})$ bị chặn khi $\mathbf{R}^{(i)} \to +\infty$; (4) thang đo có thể điều chỉnh được; (5) và hàm là đơn điệu. Độ phức tạp thời gian bổ sung do $f(\mathbf{R}^{(i)})$ mang lại là $\mathcal{O}(L^2)$ và nó nhỏ so với độ phức tạp thời gian của tự chú ý $\mathcal{O}(L^2 d)$. Mức tiêu thụ bộ nhớ bổ sung là tối thiểu, ~$\mathcal{O}(2h)$.

Thay vì các hệ số nhân, _ALiBi_ ([Press et al. 2022](https://arxiv.org/abs/2108.12409)) thêm một thành phần thiên vị không đổi vào điểm chú ý truy vấn-khóa, tỷ lệ thuận với khoảng cách theo cặp. Thiên vị này tạo ra một sở thích mạnh mẽ đối với tính gần đây và phạt các khóa ở quá xa. Các hình phạt được tăng lên với các tỷ lệ khác nhau trong các đầu khác nhau.

$$
\text{softmax}(\mathbf{q}_i \mathbf{K}^\top + \alpha_i \cdot [0, -1, -2, \dots, -(i-1)])
$$

trong đó $\alpha_i$ là một vô hướng trọng số cụ thể cho mỗi đầu. Khác với DA-transformer, $\alpha_i$ không được học mà được cố định dưới dạng một chuỗi hình học; ví dụ, đối với 8 đầu, ${\alpha_i} = {\frac{1}{2}, \frac{1}{2^2}, \dots, \frac{1}{2^8}}$. Ý tưởng tổng thể rất giống với những gì mà mã hóa vị trí tương đối nhằm giải quyết.

![Minh họa về cách ALiBi tăng cường điểm chú ý bằng một thành phần thiên vị vị trí.](/posts/transformer-family-2/ALiBi-bias.png)
_Minh họa về cách ALiBi tăng cường điểm chú ý bằng một thành phần thiên vị vị trí. (Nguồn ảnh: [Press et al. 2021](https://arxiv.org/abs/2108.12409))._

Với ALiBi, [Press et al. (2022)](https://arxiv.org/abs/2108.12409) đã huấn luyện một mô hình 1.3B trên một độ dài ngữ cảnh là 1024 trong quá trình huấn luyện và ngoại suy đến 2046 tại thời điểm suy luận.

![Các thí nghiệm ngoại suy để chạy suy luận với các Transformer có các cấu hình khác nhau, bao gồm mã hóa vị trí hình sin, mã hóa vị trí quay, mã hóa vị trí tương đối đơn giản hóa trong T5 và ALiBi. Tất cả các mô hình đã được huấn luyện với độ dài ngữ cảnh nhỏ nhưng suy luận đã được chạy cho ngữ cảnh dài hơn nhiều.](/posts/transformer-family-2/ALiBi-exp.png)
_Các thí nghiệm ngoại suy để chạy suy luận với các Transformer có các cấu hình khác nhau, bao gồm mã hóa vị trí hình sin, mã hóa vị trí quay, mã hóa vị trí tương đối đơn giản hóa trong T5 và ALiBi. Tất cả các mô hình đã được huấn luyện với độ dài ngữ cảnh nhỏ nhưng suy luận đã được chạy cho ngữ cảnh dài hơn nhiều. (Nguồn ảnh: [Press et al. 2021](https://arxiv.org/abs/2108.12409))._

## Biến nó thành lặp lại

**Universal Transformer** ([Dehghani, et al. 2019](https://arxiv.org/abs/1807.03819)) kết hợp cơ chế tự chú ý trong Transformer với cơ chế lặp lại trong RNN, nhằm mục đích tận dụng cả trường tiếp nhận toàn cục dài hạn của Transformer và các thiên vị quy nạp đã học của RNN. Thay vì đi qua một số lớp cố định, Universal Transformer điều chỉnh động số bước bằng cách sử dụng [thời gian tính toán thích ứng](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/#adaptive-computation-time-act). Nếu chúng ta cố định số bước, một Universal Transformer tương đương với một Transformer nhiều lớp với các tham số được chia sẻ giữa các lớp.

Ở một cấp độ cao, transformer vạn năng có thể được xem như một hàm đệ quy để học biểu diễn trạng thái ẩn cho mỗi token. Hàm đệ quy phát triển song song qua các vị trí của token và thông tin giữa các vị trí được chia sẻ thông qua cơ chế tự chú ý.

![Cách Universal Transformer tinh chỉnh một tập hợp các biểu diễn trạng thái ẩn lặp đi lặp lại cho mỗi vị trí một cách song song.](/posts/transformer-family-2/universal-transformer-loop.png)
_Cách Universal Transformer tinh chỉnh một tập hợp các biểu diễn trạng thái ẩn lặp đi lặp lại cho mỗi vị trí một cách song song. (Nguồn ảnh: Hình 1 trong [Dehghani, et al. 2019](https://arxiv.org/abs/1807.03819))._

Cho một chuỗi đầu vào có độ dài $L$, Universal Transformer cập nhật lặp đi lặp lại biểu diễn $\mathbf{h}^t \in \mathbb{R}^{L \times d}$ tại bước $t$ cho một số bước có thể điều chỉnh. Tại bước 0, $\mathbf{h}^0$ được khởi tạo giống như ma trận nhúng đầu vào. Tất cả các vị trí được xử lý song song trong cơ chế chú ý đa đầu và sau đó đi qua một hàm chuyển đổi lặp lại.

$$
\begin{aligned}
\mathbf{A}^t &= \text{LayerNorm}(\mathbf{h}^{t-1} + \text{MultiHeadAttention}(\mathbf{h}^{t-1} + \mathbf{P}^t) \\
\mathbf{h}^t &= \text{LayerNorm}(\mathbf{A}^{t-1} + \text{Transition}(\mathbf{A}^t))
\end{aligned}
$$

trong đó $\text{Transition}(.)$ là một [tích chập khả tách](https://arxiv.org/abs/1610.02357) hoặc một mạng nơ-ron kết nối đầy đủ bao gồm hai phép biến đổi afin theo vị trí (tức là áp dụng cho từng hàng của $\mathbf{A}^t$ riêng lẻ) + một ReLU.

Mã hóa vị trí $\mathbf{P}^t$ sử dụng [tín hiệu vị trí hình sin](#sinusoidal-positional-encoding) nhưng có thêm một chiều thời gian:

$$
\text{PE}(i, t, \delta) =
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) \oplus \sin(\frac{t}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
\cos(\frac{i}{10000^{2\delta'/d}}) \oplus \cos(\frac{t}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
\end{cases}
$$

![Một minh họa đơn giản về Universal Transformer. Bộ mã hóa và bộ giải mã chia sẻ cùng một cấu trúc lặp lại cơ bản. Nhưng bộ giải mã cũng chú ý đến biểu diễn cuối cùng của bộ mã hóa $\mathbf{h}^T$.](/posts/transformer-family-2/universal-transformer.png)
_Một minh họa đơn giản về Universal Transformer. Bộ mã hóa và bộ giải mã chia sẻ cùng một cấu trúc lặp lại cơ bản. Nhưng bộ giải mã cũng chú ý đến biểu diễn cuối cùng của bộ mã hóa $\mathbf{h}^T$. (Nguồn ảnh: Hình 2 trong [Dehghani, et al. 2019](https://arxiv.org/abs/1807.03819))_

Trong phiên bản thích ứng của Universal Transformer, số bước lặp lại $T$ được xác định động bởi [ACT](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/#adaptive-computation-time-act). Mỗi vị trí được trang bị một cơ chế dừng ACT động. Khi một khối lặp lại cho mỗi token dừng lại, nó sẽ ngừng nhận thêm các cập nhật lặp lại mà chỉ đơn giản là sao chép giá trị hiện tại sang bước tiếp theo cho đến khi tất cả các khối dừng lại hoặc cho đến khi mô hình đạt đến giới hạn bước tối đa.

# Mô hình hóa Thích ứng

Mô hình hóa thích ứng đề cập đến một cơ chế có thể điều chỉnh lượng tính toán theo các đầu vào khác nhau. Ví dụ, một số token chỉ có thể cần thông tin cục bộ và do đó yêu cầu một khoảng chú ý ngắn hơn; hoặc một số token tương đối dễ dự đoán hơn và không cần phải được xử lý qua toàn bộ ngăn xếp chú ý.

## Khoảng Chú ý Thích ứng (Adaptive Attention Span)

Một ưu điểm chính của Transformer là khả năng nắm bắt các phụ thuộc dài hạn. Tùy thuộc vào ngữ cảnh, mô hình có thể ưu tiên chú ý xa hơn vào những thời điểm khác nhau; hoặc một đầu chú ý có thể có một mô hình chú ý khác với đầu kia. Nếu khoảng chú ý có thể linh hoạt điều chỉnh độ dài của nó và chỉ nhìn xa hơn khi cần thiết, điều này sẽ giúp giảm cả chi phí tính toán và bộ nhớ để hỗ trợ kích thước ngữ cảnh tối đa lớn hơn trong mô hình.

Đây là động lực cho **Khoảng Chú ý Thích ứng (Adaptive Attention Span)**. [Sukhbaatar et al (2019)](https://arxiv.org/abs/1905.07799) đã đề xuất một cơ chế tự chú ý tìm kiếm một khoảng chú ý tối ưu. Họ đưa ra giả thuyết rằng các đầu chú ý khác nhau có thể gán điểm số khác nhau trong cùng một cửa sổ ngữ cảnh (Xem Hình 14) và do đó, khoảng chú ý tối ưu sẽ được huấn luyện riêng cho mỗi đầu.

![Hai đầu chú ý trong cùng một mô hình, A và B, gán sự chú ý khác nhau trong cùng một cửa sổ ngữ cảnh. Đầu A chú ý nhiều hơn đến các token gần đây, trong khi đầu B nhìn xa hơn vào quá khứ một cách đồng đều.](/posts/transformer-family-2/attention-per-head.png)
_Hai đầu chú ý trong cùng một mô hình, A và B, gán sự chú ý khác nhau trong cùng một cửa sổ ngữ cảnh. Đầu A chú ý nhiều hơn đến các token gần đây, trong khi đầu B nhìn xa hơn vào quá khứ một cách đồng đều. (Nguồn ảnh: [Sukhbaatar, et al. 2019](https://arxiv.org/abs/1905.07799))_

Cho token thứ $i$, chúng ta cần tính toán trọng số chú ý giữa token này và các khóa khác trong khoảng chú ý của nó có kích thước $s$:

$$
\begin{aligned}
e_{ij} &= \mathbf{q}_i {\mathbf{k}_j}^\top \\
a_{ij} &= \text{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{r=i-s}^{i-1} \exp(e_{ir})} \\
\mathbf{y}_i &= \sum_{r=i-s}^{i-1}a_{ir}\mathbf{v}_r = \sum_{r=i-s}^{i-1}a_{ir}\mathbf{x}_r\mathbf{W}^v
\end{aligned}
$$

Một _hàm mặt nạ mềm_ $m_z$ được thêm vào để kiểm soát một khoảng chú ý có thể điều chỉnh hiệu quả, ánh xạ khoảng cách giữa truy vấn và khóa thành một giá trị. $m_z$ được tham số hóa bởi $z \in [0, s]$ và $z$ cần được học:

$$
m_z(x) = \text{clip}(\frac{1}{R}(R+z-x), 0, 1)
$$

trong đó $R$ là một siêu tham số xác định độ mềm của $m_z$.

![Hàm che mềm được sử dụng trong khoảng chú ý thích ứng.](/posts/transformer-family-2/soft-masking-function.png)
_Hàm che mềm được sử dụng trong khoảng chú ý thích ứng. (Nguồn ảnh: [Sukhbaatar, et al. 2019](https://arxiv.org/abs/1905.07799).)_

Hàm mặt nạ mềm được áp dụng cho các phần tử softmax trong trọng số chú ý:

$$
a_{ij} = \frac{m_z(i-j)\exp(s_{ij})}{\sum_{r=i-s}^{i-1}m_z(i-r) \exp(s_{ir})}
$$

Trong phương trình trên, $z$ là khả vi nên được huấn luyện cùng với các phần khác của mô hình. Các tham số $z^{(i)}, i=1, \dots, h$ được học _riêng cho mỗi đầu_. Hơn nữa, hàm mất mát có thêm một hình phạt L1 trên $\sum_{i=1}^h z^{(i)}$.

Bằng cách sử dụng [Thời gian tính toán thích ứng](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/#adaptive-computation-time-act), phương pháp này có thể được cải thiện hơn nữa để có độ dài khoảng chú ý linh hoạt, thích ứng động với đầu vào hiện tại. Tham số khoảng $z_t$ của một đầu chú ý tại thời điểm $t$ là một hàm sigmoid, $z_t = S \sigma(\mathbf{v} \cdot \mathbf{x}_t +b)$, trong đó vector $\mathbf{v}$ và vô hướng thiên vị $b$ được học cùng với các tham số khác.

Trong các thí nghiệm của Transformer với khoảng chú ý thích ứng, [Sukhbaatar, et al. (2019)](https://arxiv.org/abs/1905.07799) đã tìm thấy một xu hướng chung là các lớp thấp hơn không yêu cầu khoảng chú ý rất dài, trong khi một vài đầu chú ý ở các lớp cao hơn có thể sử dụng các khoảng chú ý đặc biệt dài. Khoảng chú ý thích ứng cũng giúp giảm đáng kể số lượng FLOPS, đặc biệt là trong một mô hình lớn có nhiều lớp chú ý và độ dài ngữ cảnh lớn.

## Transformer thích ứng theo độ sâu

Tại thời điểm suy luận, tự nhiên là giả định rằng một số token dễ dự đoán hơn và do đó không cần nhiều tính toán như những token khác. Do đó, chúng ta có thể chỉ xử lý dự đoán của nó qua một số lượng lớp hạn chế để đạt được sự cân bằng tốt giữa tốc độ và hiệu suất.

Cả **Transformer Thích ứng theo Độ sâu (Depth-Adaptive Transformer)** ([Elabyad et al. 2020](https://arxiv.org/abs/1910.10073)) và **Mô hình Ngôn ngữ Thích ứng Tự tin (Confident Adaptive Language Model - CALM)** ([Schuster et al. 2022](https://arxiv.org/abs/2207.07061)) đều được thúc đẩy bởi ý tưởng này và học cách dự đoán số lượng lớp tối ưu cần thiết cho các token đầu vào khác nhau.

_Transformer thích ứng theo độ sâu_ ([Elabyad et al. 2020](https://arxiv.org/abs/1910.10073)) gắn một bộ phân loại đầu ra vào mỗi lớp để tạo ra các dự đoán thoát dựa trên các kích hoạt của lớp đó. Các ma trận trọng số của bộ phân loại có thể khác nhau cho mỗi lớp hoặc được chia sẻ giữa các lớp. Trong quá trình huấn luyện, mô hình lấy mẫu các chuỗi thoát khác nhau sao cho mô hình được tối ưu hóa với các trạng thái ẩn của các lớp khác nhau. Mục tiêu học tập kết hợp các xác suất có khả năng xảy ra được dự đoán ở các lớp khác nhau, $n=1, \dots, N$:

$$
\text{LL}^n_t = \log p(y_t \vert \mathbf{h}^n_{t-1}) \quad
\text{LL}^n = \sum_{t=1}^{\vert\mathbf{y}\vert} LL^n_t
$$

Các bộ phân loại độ sâu thích ứng cho ra một phân phối tham số $q_t$. Nó được huấn luyện với hàm mất mát cross entropy so với một phân phối oracle $q^*_t$. Bài báo đã khám phá ba cấu hình về cách học một bộ phân loại như vậy $q_t$.

![Minh họa về ba loại bộ phân loại độ sâu thích ứng.](/posts/transformer-family-2/depth-adaptive-classifier.png)
_Minh họa về ba loại bộ phân loại độ sâu thích ứng. <br/>(Nguồn ảnh: [Elabyad et al. 2020](https://arxiv.org/abs/1910.10073))._

1.  _Bộ phân loại độ sâu đặc trưng cho chuỗi_: Tất cả các token của cùng một chuỗi chia sẻ cùng một khối thoát. Nó phụ thuộc vào giá trị trung bình của biểu diễn bộ mã hóa của chuỗi. Cho một chuỗi đầu vào $\mathbf{x}$ có độ dài $L$, bộ phân loại nhận $\bar{\mathbf{x}} = \frac{1}{L} \sum_{t=1}^L \mathbf{x}_t$ làm đầu vào và cho ra một phân phối đa thức có $N$ chiều, tương ứng với $N$ lớp.

    $$
    \begin{aligned}
    q(n \vert \mathbf{x}) &=\text{softmax}(\mathbf{W}_n \bar{\mathbf{x}} + b_n) \in \mathbb{R}^N \\
    q_\text{lik}^*(\mathbf{x}, \mathbf{y}) &= \delta(\arg\max_n \text{LL}^n - \lambda n) \\
    \text{or }q_\text{corr}^*(\mathbf{x}, \mathbf{y}) &= \delta(\arg\max_n C^n - \lambda n) \text{ where }C^n = \vert\{t \vert y_t = \arg\max_y p(y \vert \mathbf{h}^n_{t-1})\}\vert \\
    \end{aligned}
    $$

    trong đó $\delta$ là [hàm delta Dirac](https://en.wikipedia.org/wiki/Dirac_delta_function) (hàm xung đơn vị) và $-\lambda n$ là một thuật ngữ chính quy hóa để khuyến khích các lối thoát ở các lớp thấp hơn. Sự thật cơ bản $q^*$ có thể được chuẩn bị theo hai cách, dựa trên khả năng tối đa $q_\text{lik}^*$ hoặc tính đúng đắn $q_\text{corr}^*$.

2.  _Bộ phân loại độ sâu đặc trưng cho token (đa thức)_: Mỗi token được giải mã bằng một khối thoát khác nhau, được dự đoán dựa trên trạng thái ẩn đầu tiên của bộ giải mã $\mathbf{h}^1_t$:

    $$
    q_t(n \vert \mathbf{x}, \mathbf{y}_{< t}) = \text{softmax}(\mathbf{W}_n \mathbf{h}^1_t + b_n)
    $$

3.  _Bộ phân loại độ sâu đặc trưng cho token (giống hình học)_: Một phân phối dự đoán thoát nhị phân được tạo cho mỗi lớp cho mỗi token, $\mathcal{X}^n_t$. Kernel RBF $\kappa(t, t') = \exp(\frac{\vert t - t' \vert^2}{\sigma})$ được sử dụng để làm mịn các dự đoán nhằm kết hợp tác động của quyết định hiện tại lên các bước thời gian trong tương lai.
    $$
    \begin{aligned}
    \mathcal{X}^n_t &= \text{sigmoid}(\mathbf{w}_n^\top \mathbf{h}^n_t + b_n)\quad \forall n \in [1, \dots, N-1] \\
    q_t(n \vert \mathbf{x}, \mathbf{y}_{< t}) &= \begin{cases}
    \mathcal{X}^n_t \prod_{n' < n} (1 - \mathcal{X}^{n'}_t) & \text{if } n < N\\
    \prod_{n' < N} (1 - \mathcal{X}^{n'}_t) & \text{otherwise}
    \end{cases} \\
    q_\text{lik}^*(\mathbf{x}, \mathbf{y}) &= \delta(\arg\max_n \widetilde{\text{LL}}^n_t - \lambda n) \text{ where } \widetilde{\text{LL}}^n_t = \sum_{t'=1}^{\vert\mathbf{y}\vert}\kappa(t, t') LL^n_{t'} \\
    \text{or }q_\text{cor}^*(\mathbf{x}, \mathbf{y}) &= \delta(\arg\max_n \tilde{C}_t^n - \lambda n) \text{ where }C_t^n = \mathbb{1}[y_t = \arg\max_y p(y \vert \mathbf{h}^n_{t-1})],\; \tilde{C}^n_t = \sum_{t'=1}^{\vert\mathbf{y}\vert}\kappa(t, t') C^n_{t'} \\
    \end{aligned}
    $$

Tại thời điểm suy luận, ngưỡng tin cậy để đưa ra quyết định thoát cần được hiệu chỉnh. Transformer thích ứng theo độ sâu tìm thấy một ngưỡng như vậy trên một tập hợp xác thực thông qua tìm kiếm lưới. _CALM_ ([Schuster et al. 2022](https://arxiv.org/abs/2207.07061)) đã áp dụng khuôn khổ Học rồi Kiểm tra (LTT) ([Angelopoulos et al. 2021](https://arxiv.org/abs/2110.01052)) để xác định một tập hợp con các ngưỡng hợp lệ và chọn giá trị tối thiểu làm ngưỡng cho suy luận. Ngoài việc huấn luyện bộ phân loại thoát cho mỗi lớp, CALM cũng đã khám phá các phương pháp khác để dự đoán độ sâu thích ứng, bao gồm các phản hồi softmax (tức là, sự khác biệt giữa hai đầu ra softmax hàng đầu) và độ bão hòa trạng thái ẩn (tức là, $\cos(\mathbf{h}^n_t, \mathbf{h}^{n+1}_t)$) làm điểm tin cậy cho các quyết định thoát. Họ phát hiện ra rằng các phản hồi softmax mang lại sự tăng tốc suy luận tốt nhất.

# Chú ý hiệu quả (Efficient Attention)

Chi phí tính toán và bộ nhớ của vanilla Transformer tăng theo cấp số nhân với độ dài chuỗi và do đó rất khó để áp dụng nó cho các chuỗi rất dài. Nhiều cải tiến hiệu quả cho kiến trúc Transformer có liên quan đến mô-đun tự chú ý - làm cho nó rẻ hơn, nhỏ hơn hoặc chạy nhanh hơn. Xem bài báo khảo sát về _Transformer hiệu quả_ ([Tay et al. 2020](https://arxiv.org/abs/2009.06732)).

## Các mẫu chú ý thưa (Sparse Attention Patterns)

### Ngữ cảnh Cục bộ Cố định

Một sự thay đổi đơn giản để làm cho tự chú ý ít tốn kém hơn là giới hạn khoảng chú ý của mỗi token chỉ trong ngữ cảnh **cục bộ**, để tự chú ý phát triển tuyến tính với độ dài chuỗi.

Ý tưởng này được giới thiệu bởi **Image Transformer** ([Parmer, et al 2018](https://arxiv.org/abs/1802.05751)), trong đó xây dựng mô hình tạo ảnh dưới dạng mô hình chuỗi sử dụng kiến trúc transformer mã hóa-giải mã:

- Bộ mã hóa tạo ra một biểu diễn có ngữ cảnh, theo từng kênh pixel của ảnh nguồn;
- Sau đó, bộ giải mã tự động tạo ra một ảnh đầu ra, một kênh cho mỗi pixel tại mỗi bước thời gian.

Hãy ký hiệu biểu diễn của pixel hiện tại sẽ được tạo ra là truy vấn $\mathbf{q}$. Các vị trí khác có biểu diễn sẽ được sử dụng để tính toán $\mathbf{q}$ là các vector khóa $\mathbf{k}_1, \mathbf{k}_2, \dots$ và chúng cùng nhau tạo thành một ma trận bộ nhớ $\mathbf{M}$. Phạm vi của $\mathbf{M}$ xác định cửa sổ ngữ cảnh cho truy vấn pixel $\mathbf{q}$.

Image Transformer đã giới thiệu hai loại $\mathbf{M}$ cục bộ hóa, như được minh họa dưới đây.

![Minh họa về khoảng chú ý 1D và 2D cho đầu vào hình ảnh trong Image Transformer. Đường màu đen đánh dấu một khối truy vấn và đường viền màu xanh lam phác thảo khoảng chú ý thực tế cho pixel q.](/posts/transformer-family-2/image-transformer-attention.png)
_Minh họa về khoảng chú ý 1D và 2D cho đầu vào hình ảnh trong Image Transformer. Đường màu đen đánh dấu một khối truy vấn và đường viền màu xanh lam phác thảo khoảng chú ý thực tế cho pixel q. (Nguồn ảnh: Hình 2 trong [Parmer et al, 2018](https://arxiv.org/abs/1802.05751))_

1.  _Chú ý Cục bộ 1D_: Ảnh đầu vào được làm phẳng theo thứ tự [quét raster](https://en.wikipedia.org/wiki/Raster_scan#Scanning_pattern), tức là từ trái sang phải và từ trên xuống dưới. Ảnh đã được tuyến tính hóa sau đó được chia thành các khối truy vấn không chồng chéo. Cửa sổ ngữ cảnh bao gồm các pixel trong cùng một khối truy vấn với $\mathbf{q}$ và một số lượng pixel bổ sung cố định được tạo ra trước khối truy vấn này.

2.  _Chú ý Cục bộ 2D_: Ảnh được chia thành nhiều khối truy vấn hình chữ nhật không chồng chéo. Pixel truy vấn có thể chú ý đến tất cả các pixel khác trong cùng một khối bộ nhớ. Để đảm bảo rằng pixel ở góc trên cùng bên trái cũng có thể có một cửa sổ ngữ cảnh hợp lệ, khối bộ nhớ được mở rộng tương ứng lên trên, sang trái và sang phải một lượng cố định.

### Ngữ cảnh có bước nhảy (Strided Context)

**Sparse Transformer** ([Child et al., 2019](https://arxiv.org/abs/1904.10509)) đã giới thiệu _tự chú ý phân tích nhân tử_, thông qua phân tích nhân tử ma trận thưa, giúp có thể huấn luyện các mạng chú ý dày đặc với hàng trăm lớp trên các chuỗi có độ dài lên đến 16.384, điều mà nếu không thì sẽ không thể thực hiện được trên phần cứng hiện đại.

Cho một tập hợp các mẫu kết nối chú ý $\mathcal{S} = \{S_1, \dots, S_n\}$, trong đó mỗi $S_i$ ghi lại một tập hợp các vị trí khóa mà vector truy vấn thứ $i$ chú ý đến.

$$
\begin{aligned}
\text{Attend}(\mathbf{X}, \mathcal{S}) &= \Big( a(\mathbf{x}_i, S_i) \Big)_{i \in \{1, \dots, L\}} \\
\text{ where } a(\mathbf{x}_i, S_i) &= \text{softmax}\Big(\frac{(\mathbf{x}_i \mathbf{W}^q)(\mathbf{x}_j \mathbf{W}^k)_{j \in S_i}^\top}{\sqrt{d_k}}\Big) (\mathbf{x}_j \mathbf{W}^v)_{j \in S_i}
\end{aligned}
$$

Lưu ý rằng mặc dù kích thước của $S_i$ không cố định, $a(\mathbf{x}_i, S_i)$ luôn có kích thước $d_v$ và do đó $\text{Attend}(\mathbf{X}, \mathcal{S}) \in \mathbb{R}^{L \times d_v}$.

Trong các mô hình tự hồi quy, một khoảng chú ý được định nghĩa là $S_i = \{j: j \leq i\}$ vì nó cho phép mỗi token chú ý đến tất cả các vị trí trong quá khứ.

Trong tự chú ý phân tích nhân tử, tập hợp $S_i$ được phân tách thành một _cây_ phụ thuộc, sao cho đối với mọi cặp $(i, j)$ trong đó $j \leq i$, có một đường dẫn nối $i$ trở lại $j$ và $i$ có thể chú ý đến $j$ một cách trực tiếp hoặc gián tiếp.

Cụ thể hơn, tập hợp $S_i$ được chia thành $p$ tập hợp con _không chồng chéo_, trong đó tập hợp con thứ $m$ được ký hiệu là $A^{(m)}_i \subset S_i, m = 1,\dots, p$. Do đó, đường đi giữa vị trí đầu ra $i$ và bất kỳ $j$ nào có độ dài tối đa là $p + 1$. Ví dụ, nếu $(j, a, b, c, \dots, i)$ là một đường đi của các chỉ số giữa $i$ và $j$, chúng ta sẽ có $j \in A_a^{(1)}, a \in A_b^{(2)}, b \in A_c^{(3)}, \dots$, và cứ thế tiếp tục.

**Chú ý Phân tích nhân tử Thưa (Sparse Factorized Attention)**

Sparse Transformer đã đề xuất hai loại chú ý phân tích nhân tử. Các khái niệm này dễ hiểu hơn khi được minh họa trong Hình 10 với các ví dụ về đầu vào hình ảnh 2D.

![Hàng trên minh họa các mẫu kết nối chú ý trong (a) Transformer, (b) Sparse Transformer với chú ý có bước nhảy, và (c) Sparse Transformer với chú ý cố định. Hàng dưới chứa các ma trận kết nối tự chú ý tương ứng. Lưu ý rằng hàng trên và hàng dưới không cùng tỷ lệ.](/posts/transformer-family-2/sparse-attention.png)
_Hàng trên minh họa các mẫu kết nối chú ý trong (a) Transformer, (b) Sparse Transformer với chú ý có bước nhảy, và (c) Sparse Transformer với chú ý cố định. Hàng dưới chứa các ma trận kết nối tự chú ý tương ứng. Lưu ý rằng hàng trên và hàng dưới không cùng tỷ lệ. (Nguồn ảnh: [Child et al., 2019](https://arxiv.org/abs/1904.10509) + một vài chú thích bổ sung.)_

1.  Chú ý _có bước nhảy (strided)_ với bước nhảy $\ell \sim \sqrt{n}$. Điều này hoạt động tốt với dữ liệu hình ảnh vì cấu trúc phù hợp với các bước nhảy. Trong trường hợp hình ảnh, mỗi pixel sẽ chú ý đến tất cả các pixel $\ell$ trước đó theo thứ tự quét raster (tự nhiên bao phủ toàn bộ chiều rộng của hình ảnh) và sau đó các pixel đó sẽ chú ý đến các pixel khác trong cùng một cột (được xác định bởi một tập hợp con kết nối chú ý khác).

    $$
    \begin{aligned}
    A_i^{(1)} &= \{ t, t+1, \dots, i\} \text{, where } t = \max(0, i - \ell) \\
    A_i^{(2)} &= \{j: (i-j) \mod \ell = 0\}
    \end{aligned}
    $$

2.  Chú ý _cố định_. Một tập hợp nhỏ các token tóm tắt các vị trí trước đó và truyền thông tin đó đến tất cả các vị trí trong tương lai.
    $$
    \begin{aligned}
    A_i^{(1)} &= \{j: \lfloor \frac{j}{\ell} \rfloor = \lfloor \frac{i}{\ell} \rfloor \} \\
    A_i^{(2)} &= \{j: j \mod \ell \in \{\ell-c, \dots, \ell-1\} \}
    \end{aligned}
    $$
    trong đó $c$ là một siêu tham số. Nếu $c=1$, nó hạn chế biểu diễn trong khi nhiều thứ phụ thuộc vào một vài vị trí. Bài báo đã chọn $c\in \{ 8, 16, 32 \}$ cho $\ell \in \{ 128, 256 \}$.

**Sử dụng Tự chú ý Phân tích nhân tử trong Transformer**

Có ba cách để sử dụng các mẫu chú ý phân tích nhân tử thưa trong kiến trúc Transformer:

1.  Một loại chú ý cho mỗi khối dư và sau đó xen kẽ chúng,
    $\text{attn}(\mathbf{X}) = \text{Attend}(\mathbf{X}, A^{(n \mod p)}) \mathbf{W}^o$, trong đó $n$ là chỉ số của khối dư hiện tại.
2.  Thiết lập một đầu duy nhất chú ý đến các vị trí mà tất cả các đầu phân tích nhân tử chú ý đến,
    $\text{attn}(\mathbf{X}) = \text{Attend}(\mathbf{X}, \cup_{m=1}^p A^{(m)}) \mathbf{W}^o $.
3.  Sử dụng cơ chế chú ý đa đầu, nhưng khác với vanilla Transformer, mỗi đầu có thể áp dụng một mẫu được trình bày ở trên, 1 hoặc 2. → Tùy chọn này thường hoạt động tốt nhất.

Sparse Transformer cũng đã đề xuất một loạt các thay đổi để có thể huấn luyện Transformer lên đến hàng trăm lớp, bao gồm kiểm tra điểm gradient, tính toán lại các lớp chú ý & FF trong quá trình truyền ngược, huấn luyện độ chính xác hỗn hợp, triển khai hiệu quả theo khối thưa, v.v. Vui lòng xem [bài báo](https://arxiv.org/abs/1904.10509) để biết thêm chi tiết hoặc bài viết trước của tôi về [các kỹ thuật để mở rộng quy mô huấn luyện mô hình](https://lilianweng.github.io/posts/2021-09-25-train-large/).

**Chú ý theo khối (Blockwise Attention)** ([Qiu et al. 2019](https://arxiv.org/abs/1911.02972)) giới thiệu một _ma trận khối thưa_ để chỉ cho phép mỗi token chú ý đến một tập hợp nhỏ các token khác. Mỗi ma trận chú ý có kích thước $L \times L$ được chia thành $n \times n$ khối nhỏ hơn có kích thước $\frac{L}{n}\times\frac{L}{n}$ và một ma trận khối thưa $\mathbf{M} \in \{0, 1\}^{L \times L}$ được định nghĩa bởi một hoán vị $\pi$ của ${1, \dots, n}$, ghi lại chỉ số cột cho mỗi hàng trong ma trận khối.

$$
\begin{aligned}
\text{attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) &= \text{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} \odot \mathbf{M}\Big)\mathbf{V} \\
(\mathbf{A} \odot \mathbf{M})_{ij} &= \begin{cases}
A_{ij} & \text{if }M_{ij} = 1 \\
-\infty & \text{if }M_{ij} = 0 \\
\end{cases} \\
\text{where } M_{ij} &= \begin{cases}
1 & \text{if }\pi\big(\lfloor\frac{(i-1)n}{L} + 1\rfloor\big) = \lfloor\frac{(j-1)n}{L} + 1\rfloor \\
0 & \text{otherwise}
\end{cases}
\end{aligned}
$$

Việc triển khai thực tế của Chú ý theo khối chỉ lưu trữ QKV dưới dạng ma trận khối, mỗi ma trận có kích thước $n\times n$:

$$
\text{Blockwise-attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \begin{bmatrix}
\text{softmax}\big(\frac{\hat{\mathbf{q}}_1\hat{\mathbf{k}}_{\pi(1)}^\top}{\sqrt{d}} \Big)\hat{\mathbf{v}}_{\pi(1)} \\
\vdots \\
\text{softmax}\big(\frac{\hat{\mathbf{q}}_n\hat{\mathbf{k}}_{\pi(n)}^\top}{\sqrt{d}} \odot \Big)\hat{\mathbf{v}}_{\pi(n)} \\
\end{bmatrix}
$$

trong đó $\hat{\mathbf{q}}_i$, $\hat{\mathbf{k}}_i$ và $\hat{\mathbf{v}}_i$ lần lượt là hàng thứ $i$ trong ma trận khối QKV. Mỗi $\mathbf{q}_i\mathbf{k}_{\pi(i)}^\top, \forall i = 1, \dots, n$ có kích thước $\frac{N}{n}\times\frac{N}{n}$ và do đó Chú ý theo khối có thể giảm độ phức tạp bộ nhớ của ma trận chú ý từ $\mathcal{O}(L^2)$ xuống $\mathcal{O}(\frac{L}{n}\times\frac{L}{n} \times n) = \mathcal{O}(L^2/n)$.

### Kết hợp Ngữ cảnh Cục bộ và Toàn cục

Các mô hình **ETC** (_Extended Transformer Construction_; [Ainslie et al. 2019](https://aclanthology.org/2020.emnlp-main.19/)), **Longformer** ([Beltagy et al. 2020](https://arxiv.org/abs/2004.05150)) và **Big Bird** ([Zaheer et al. 2020](https://arxiv.org/abs/2007.14062)) kết hợp cả ngữ cảnh cục bộ và toàn cục khi xây dựng ma trận chú ý. Tất cả các mô hình này đều có thể được khởi tạo từ các mô hình được huấn luyện trước hiện có.

**Chú ý Toàn cục-Cục bộ (Global-Local Attention)** của _ETC_ ([Ainslie et al. 2019](https://aclanthology.org/2020.emnlp-main.19/)) nhận hai đầu vào, (1) đầu vào dài $\mathbf{x}^l$ có kích thước $n_l$, là chuỗi đầu vào thông thường, và (2) đầu vào toàn cục $\mathbf{x}^g$ có kích thước $n_g$, chứa một số lượng nhỏ hơn các token phụ trợ, $n_g \ll n_l$. Do đó, chú ý được chia thành bốn thành phần dựa trên chú ý có hướng giữa hai đầu vào này: g2g, g2l, l2g và l2l. Vì phần chú ý l2l có thể rất lớn, nó được giới hạn trong một khoảng chú ý có kích thước cố định với bán kính $w$ (tức là khoảng chú ý cục bộ) và ma trận l2l có thể được định hình lại thành $n_l \times (2w+1)$.

ETC sử dụng bốn ma trận nhị phân để xử lý các đầu vào có cấu trúc, $\mathbf{M}^{g2g}$, $\mathbf{M}^{g2l}$, $\mathbf{M}^{l2g}$ và $\mathbf{M}^{l2l}$. Ví dụ, mỗi phần tử $z^g_i \in \mathbb{R}^d$ trong đầu ra chú ý $z^g = (z^g_1, \dots, z^g_{n_g})$ cho phần chú ý g2g được định dạng như sau:

$$
\begin{aligned}
a^{g2g}_{ij} = \frac{1}{\sqrt{d}} x^g_i \mathbf{W}^Q (x^g_j \mathbf{W}^K + P^K_{ij})^\top - (1- M^{g2g}_{ij})C \\
A^{g2g}_{ij} = \frac{\exp(a^{g2g}_{ij})}{\sum_{k=1}^{n_g} \exp(a^{g2g}_{ik})} \quad
z^g_i = \sum^{n_g}_{j=1} A^{g2g}_{ij} x^g_j \mathbf{W}^V
\end{aligned}
$$

trong đó $P^K_{ij}$ là một vector có thể học được để mã hóa vị trí tương đối và $C$ là một hằng số rất lớn ($C=10000$ trong bài báo) để bù trừ bất kỳ trọng số chú ý nào khi mặt nạ bị tắt.

![Các mẫu chú ý của ETC, Longformer và Big Bird.](/posts/transformer-family-2/combined-attention.png)
_Các mẫu chú ý của ETC, Longformer và Big Bird._

Một cập nhật nữa trong ETC là kết hợp một tác vụ CPC (mã hóa dự đoán tương phản) sử dụng [mất mát NCE](https://lilianweng.github.io/posts/2021-05-31-contrastive/#nce) vào giai đoạn huấn luyện trước, bên cạnh tác vụ [MLM](https://lilianweng.github.io/posts/2019-01-31-lm/#MLM): Biểu diễn của một câu nên tương tự như biểu diễn của ngữ cảnh xung quanh nó khi câu này bị che.

Đầu vào toàn cục $\mathbf{x}^g$ cho ETC được xây dựng như sau: Giả sử có một số đoạn trong các đầu vào dài (ví dụ: theo câu), mỗi đoạn được gắn với một token phụ để học các đầu vào toàn cục. [Mã hóa vị trí tương đối](#relative-position-encoding) được sử dụng để đánh dấu các token đoạn toàn cục bằng vị trí của token. Việc che cứng theo một hướng (tức là, các token trước và sau được dán nhãn khác nhau) đã được chứng minh là mang lại lợi ích về hiệu suất trong một số bộ dữ liệu.

Mẫu chú ý trong Longformer bao gồm ba thành phần:

1.  _Chú ý cục bộ_: Tương tự như ETC, chú ý cục bộ được kiểm soát bởi một cửa sổ trượt có kích thước cố định $w$;
2.  _Chú ý toàn cục của các token được chọn trước_: Longformer có một vài token được chọn trước (ví dụ: token `[CLS]`) được gán một khoảng chú ý toàn cục, tức là chú ý đến tất cả các token khác trong chuỗi đầu vào.
3.  _Chú ý giãn nở_: Cửa sổ trượt giãn nở có kích thước cố định $r$ và các khoảng trống có kích thước giãn nở $d$, tương tự như Sparse Transformer;

_Big Bird_ khá giống với Longformer, được trang bị cả chú ý cục bộ và một vài token được chọn trước có khoảng chú ý toàn cục, nhưng Big Bird thay thế chú ý giãn nở bằng một cơ chế mới trong đó tất cả các token chú ý đến một tập hợp các token ngẫu nhiên. Thiết kế này được thúc đẩy bởi thực tế là mẫu chú ý có thể được xem như một [đồ thị có hướng](https://en.wikipedia.org/wiki/Directed_graph) và một [đồ thị ngẫu nhiên](https://en.wikipedia.org/wiki/Random_graph) có thuộc tính là thông tin có thể chảy nhanh chóng giữa bất kỳ cặp nút nào.

_Longformer_ sử dụng kích thước cửa sổ nhỏ hơn ở các lớp thấp hơn và kích thước cửa sổ lớn hơn ở các lớp cao hơn. Các nghiên cứu loại bỏ đã cho thấy rằng thiết lập này hoạt động tốt hơn so với cấu hình ngược hoặc cấu hình kích thước cố định. Các lớp thấp hơn không có cửa sổ trượt giãn nở để học cách sử dụng ngữ cảnh cục bộ ngay lập tức tốt hơn. Longformer cũng có một quy trình huấn luyện theo giai đoạn, trong đó ban đầu mô hình được huấn luyện với kích thước cửa sổ nhỏ để học từ ngữ cảnh cục bộ và sau đó ở các giai đoạn huấn luyện tiếp theo, kích thước cửa sổ được tăng lên và tốc độ học giảm đi.

## Chú ý dựa trên nội dung (Content-based Attention)

Các cải tiến được đề xuất bởi **Reformer** ([Kitaev, et al. 2020](https://arxiv.org/abs/2001.04451)) nhằm giải quyết các vấn đề sau trong vanilla Transformer:

- Độ phức tạp thời gian và bộ nhớ bậc hai trong mô-đun tự chú ý.
- Bộ nhớ trong một mô hình có $N$ lớp lớn hơn $N$ lần so với một mô hình một lớp vì chúng ta cần lưu trữ các kích hoạt để lan truyền ngược.
- Các lớp FF trung gian thường khá lớn.

Reformer đã đề xuất hai thay đổi chính:

1.  Thay thế chú ý tích vô hướng bằng _chú ý băm nhạy cảm với vị trí (LSH)_, giảm độ phức tạp từ $\mathcal{O}(L^2)$ xuống $\mathcal{O}(L\log L)$.
2.  Thay thế các khối dư chuẩn bằng _các lớp dư có thể đảo ngược_, cho phép lưu trữ các kích hoạt chỉ một lần trong quá trình huấn luyện thay vì $N$ lần (tức là tỷ lệ thuận với số lượng lớp).

<a id="LSH" ></a>**Chú ý băm nhạy cảm với vị trí (Locality-Sensitive Hashing Attention)**

Trong phần $\mathbf{Q} \mathbf{K}^\top$ của [công thức chú ý](#attention-and-self-attention), chúng ta chỉ quan tâm đến các phần tử lớn nhất vì chỉ các phần tử lớn mới đóng góp nhiều sau softmax. Đối với mỗi truy vấn $\mathbf{q}_i \in \mathbf{Q}$, chúng ta đang tìm kiếm các vector hàng trong $\mathbf{K}$ gần nhất với $\mathbf{q}_i$. Để tìm nhanh các hàng xóm gần nhất trong không gian có chiều cao, Reformer kết hợp [Băm nhạy cảm với vị trí (LSH)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) vào cơ chế chú ý của nó.

Một lược đồ băm $x \mapsto h(x)$ là _nhạy cảm với vị trí_ nếu nó bảo toàn thông tin khoảng cách giữa các điểm dữ liệu, sao cho các vector gần nhau có các giá trị băm tương tự nhau trong khi các vector xa nhau có các giá trị băm rất khác nhau. Reformer áp dụng một lược đồ băm như vậy: cho một ma trận ngẫu nhiên cố định $\mathbf{R} \in \mathbb{R}^{d \times b/2}$ (trong đó $b$ là một siêu tham số), hàm băm là $h(x) = \arg\max([xR; −xR])$.

![Minh họa về chú ý băm nhạy cảm với vị trí (LSH).](/posts/transformer-family-2/LSH-attention-matrix.png)
_Minh họa về chú ý băm nhạy cảm với vị trí (LSH). (Nguồn ảnh: phần bên phải của Hình 1 trong [Kitaev, et al. 2020](https://arxiv.org/abs/2001.04451))._

Trong chú ý LSH, một truy vấn chỉ có thể chú ý đến các vị trí trong cùng một thùng băm, $S_i = \{j: h(\mathbf{q}_i) = h(\mathbf{k}_j)\}$. Nó được thực hiện theo quy trình sau, như được minh họa trong Hình 20:

- (a) Ma trận chú ý cho chú ý đầy đủ thường là thưa.
- (b) Sử dụng LSH, chúng ta có thể sắp xếp các khóa và truy vấn để được căn chỉnh theo các thùng băm của chúng.
- (c) Đặt $\mathbf{Q} = \mathbf{K}$ (chính xác là $\mathbf{k}_j = \mathbf{q}_j / |\mathbf{q}_j|$), để có số lượng khóa và truy vấn bằng nhau trong một thùng, dễ dàng cho việc xử lý theo lô. Điều thú vị là cấu hình "QK chung" này không ảnh hưởng đến hiệu suất của Transformer.
- (d) Áp dụng xử lý theo lô trong đó các khối gồm $m$ truy vấn liên tiếp được nhóm lại với nhau.

![Chú ý LSH bao gồm 4 bước: chia thùng, sắp xếp, chia khối và tính toán chú ý.](/posts/transformer-family-2/LSH-attention.png)
_Chú ý LSH bao gồm 4 bước: chia thùng, sắp xếp, chia khối và tính toán chú ý. (Nguồn ảnh: phần bên trái của Hình 1 trong [Kitaev, et al. 2020](https://arxiv.org/abs/2001.04451))._

**Mạng dư có thể đảo ngược (Reversible Residual Network)**

Một cải tiến khác của Reformer là sử dụng _các lớp dư có thể đảo ngược_ ([Gomez et al. 2017](https://arxiv.org/abs/1707.04585)). Động lực cho mạng dư có thể đảo ngược là thiết kế kiến trúc sao cho các kích hoạt ở bất kỳ lớp nào cũng có thể được phục hồi từ các kích hoạt ở lớp tiếp theo, chỉ sử dụng các tham số của mô hình. Do đó, chúng ta có thể tiết kiệm bộ nhớ bằng cách tính toán lại kích hoạt trong quá trình lan truyền ngược thay vì lưu trữ tất cả các kích hoạt.

Cho một lớp $x \mapsto y$, lớp dư thông thường thực hiện $y = x + F(x)$, nhưng lớp có thể đảo ngược chia cả đầu vào và đầu ra thành các cặp $(x_1, x_2) \mapsto (y_1, y_2)$ và sau đó thực hiện các thao tác sau:

$$
y_1 = x_1 + F(x_2),\; y_2 = x_2 + G(y_1)
$$

và việc đảo ngược rất dễ dàng:

$$
x_2 = y_2 - G(y_1), \; x_1 = y_1 − F(x_2)
$$

Reformer áp dụng cùng một ý tưởng cho Transformer bằng cách kết hợp các lớp chú ý ($F$) và các lớp truyền thẳng ($G$) trong một khối mạng có thể đảo ngược:

$$
Y_1 = X_1 + \text{Attention}(X_2), \; Y_2 = X_2 + \text{FeedForward}(Y_1)
$$

Bộ nhớ có thể được giảm thêm bằng cách chia nhỏ tính toán truyền thẳng:

$$
Y_2 = [Y_2^{(1)}; \dots; Y_2^{(c)}] = [X_2^{(1)} + \text{FeedForward}(Y_1^{(1)}); \dots; X_2^{(c)} + \text{FeedForward}(Y_1^{(c)})]
$$

Transformer có thể đảo ngược kết quả không cần phải lưu trữ kích hoạt ở mọi lớp.

**Transformer định tuyến (Routing Transformer)** ([Roy et al. 2021](https://arxiv.org/abs/2003.05997)) cũng được xây dựng dựa trên việc phân cụm các khóa và truy vấn dựa trên nội dung. Thay vì sử dụng một hàm băm tĩnh như LSH, nó sử dụng phân cụm $k$-means trực tuyến và kết hợp nó với chú ý thưa cục bộ, theo thời gian để giảm độ phức tạp của chú ý từ $O(L^2)$ xuống $O(L^{1.5})$.

Trong chú ý định tuyến, cả khóa và truy vấn đều được phân cụm bằng phương pháp phân cụm $k$-means và cùng một tập hợp các tâm cụm $\boldsymbol{\mu} = (\mu_1, \dots, \mu_k) \in \mathbb{R}^{k \times d}$. Các truy vấn được định tuyến đến các khóa được gán cho cùng một tâm cụm. Tổng độ phức tạp là $O(Lkd + L^2d/k)$, trong đó $O(Lkd)$ là để chạy các phép gán cụm và $O(L^2d/k)$ là để tính toán chú ý. Các tâm cụm được cập nhật bằng EMA (trung bình động hàm mũ) sử dụng tất cả các khóa và truy vấn liên quan.

Trong các thí nghiệm cho Transformer Định tuyến, một số cấu hình tốt nhất chỉ bật chú ý định tuyến ở hai lớp cuối cùng của mô hình và một nửa số đầu chú ý, trong khi nửa còn lại sử dụng chú ý cục bộ. Họ cũng quan sát thấy rằng chú ý cục bộ là một đường cơ sở khá mạnh và cửa sổ chú ý lớn hơn luôn dẫn đến kết quả tốt hơn.

## Chú ý Hạng thấp (Low-Rank Attention)

**Linformer** ([Wang et al. 2020](https://arxiv.org/abs/2006.04768)) xấp xỉ ma trận chú ý đầy đủ bằng một ma trận _hạng thấp_, giảm độ phức tạp thời gian và không gian thành _tuyến tính_. Thay vì sử dụng SVD tốn kém để xác định phân rã hạng thấp, Linformer thêm hai phép chiếu tuyến tính $\mathbf{E}_i, \mathbf{F}_i \in \mathbb{R}^{L \times k}$ cho các ma trận khóa và giá trị tương ứng, giảm kích thước của chúng từ $L \times d$ xuống $k \times d$. Miễn là $k \ll L$, bộ nhớ chú ý có thể được giảm đáng kể.

$$
\begin{aligned}
\overline{\text{head}}_i
&= \text{attn}(\mathbf{X}_q\mathbf{W}^q_i, \mathbf{E}_i\mathbf{X}_k\mathbf{W}^k_i, \mathbf{F}_i\mathbf{X}_v\mathbf{W}^v_i) \\
&= \underbrace{\text{softmax}\Big( \frac{\mathbf{X}_q\mathbf{W}^q_i (\mathbf{E}_i \mathbf{X}_k\mathbf{W}^k_i)^\top}{\sqrt{d}} \Big)}_{\text{ma trận chú ý hạng thấp }\bar{A} \in \mathbb{R}^{k \times d}} \mathbf{F}_i \mathbf{X}_v\mathbf{W}^v_i
\end{aligned}
$$

Các kỹ thuật bổ sung có thể được áp dụng để cải thiện hơn nữa hiệu quả của Linformer:

- Chia sẻ tham số giữa các lớp chiếu, chẳng hạn như chia sẻ theo đầu, khóa-giá trị và theo lớp (trên tất cả các lớp).
- Sử dụng các $k$ khác nhau ở các lớp khác nhau, vì các đầu ở các lớp cao hơn có xu hướng có phân phối lệch hơn (hạng thấp hơn) và do đó chúng ta có thể sử dụng $k$ nhỏ hơn ở các lớp cao hơn.
- Sử dụng các loại chiếu khác nhau; ví dụ: gộp trung bình/tối đa, lớp tích chập với kernel và bước nhảy $L/k$.

![Kiến trúc và hiệu suất của Linformer](/posts/transformer-family-2/linformer.png)
_(Trái) Linformer có hai lớp chiếu được thêm vào cho các khóa và giá trị. (Phải) Biểu đồ thời gian suy luận là một hàm của độ dài chuỗi. (Nguồn ảnh: [Wang et al. 2020](https://arxiv.org/abs/2006.04768))._

**Chú ý Đặc trưng Ngẫu nhiên (Random Feature Attention - RFA)** ([Peng et al. 2021](https://arxiv.org/abs/2103.02143)) dựa vào _các phương pháp đặc trưng ngẫu nhiên_ ([Rahimi & Recht, 2007](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)) để xấp xỉ phép toán softmax trong tự chú ý bằng các bản đồ đặc trưng hạng thấp nhằm đạt được độ phức tạp thời gian và không gian tuyến tính. **Performers** ([Choromanski et al. 2021](https://arxiv.org/abs/2009.14794)) cũng áp dụng chú ý đặc trưng ngẫu nhiên với các cải tiến trong việc xây dựng kernel để giảm thêm lỗi xấp xỉ của kernel.

Định lý chính đằng sau RFA đến từ [Rahimi & Recht, 2007](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf):

> Gọi $\phi: \mathbb{R}^d \to \mathbb{R}^{2D}$ là một phép biến đổi phi tuyến:
>
> $$
> \phi(\mathbf{x}) = \frac{1}{\sqrt{D}}[\sin(\mathbf{w}_1^\top \mathbf{x}), \dots, \sin(\mathbf{w}_D^\top \mathbf{x}), \cos(\mathbf{w}_1^\top \mathbf{x}), \dots, \cos(\mathbf{w}_D^\top \mathbf{x})]^\top
> $$
>
> Khi các vector ngẫu nhiên $d$-chiều $\mathbf{w}_i$ là i.i.d. từ $\mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}_d)$,
>
> $$
> \mathbb{E}_{\mathbf{w}_i} [\phi(\mathbf{x}) \cdot \phi(\mathbf{y})] = \exp(-\frac{\| \mathbf{x} - \mathbf{y} \|^2}{2\sigma^2})
> $$

Một ước tính không chệch của $\exp(\mathbf{x} \cdot \mathbf{y})$ là:

$$
\begin{aligned}
\exp(\mathbf{x} \cdot \mathbf{y} / \sigma^2)
&= \exp(\frac{1}{2\sigma^2}(\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - \|\mathbf{x} - \mathbf{y}\|^2) \\
&= \exp(\frac{\|\mathbf{x}\|^2}{2\sigma^2}) \exp(\frac{\|\mathbf{y}\|^2}{2\sigma^2}) ( - \frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\sigma^2}) \\
&\approx \exp(\frac{\|\mathbf{x}\|^2}{2\sigma^2}) \exp(\frac{\|\mathbf{y}\|^2}{2\sigma^2})\;\phi(\mathbf{x})\cdot\phi(\mathbf{y}) \\
&= \exp(\frac{1}{\sigma^2})\;\phi(\mathbf{x})\cdot\phi(\mathbf{y}) & \text{; các vector đơn vị}
\end{aligned}
$$

Sau đó, chúng ta có thể viết hàm chú ý như sau, trong đó $\otimes$ là phép toán tích ngoài và $\sigma^2$ là nhiệt độ:

$$
\begin{aligned}
\text{attn}(\mathbf{q}_t, \{\mathbf{k}_i\}, \{\mathbf{v}_i\})
&= \sum_i \frac{\exp(\mathbf{q}_t\cdot\mathbf{k}_i/\sigma^2)}{\sum_j \exp(\mathbf{q}_t\cdot\mathbf{k}_j/\sigma^2)}\mathbf{v}_i^\top
\approx \sum_i \frac{\phi(\mathbf{q}_t)\phi(\mathbf{k}_i)\mathbf{v}_i^\top}{\sum_j \phi(\mathbf{q}_t)\phi(\mathbf{k}_j)} \\
&= \color{green}{\frac{\phi(\mathbf{q}_t)^\top \sum_i \phi(\mathbf{k}_i)\otimes\mathbf{v}_i}{\phi(\mathbf{q}_t)^\top \sum_j \phi(\mathbf{k}_j)}
= \text{RFA}(\mathbf{q}_t, \{\mathbf{k}_i\}, \{\mathbf{v}_i\})}
\end{aligned}
$$

![Thứ tự Tính toán RFA](/posts/transformer-family-2/RFA.png)
_(Trái) Thứ tự tính toán cho phép toán softmax mặc định. (Phải) Thứ tự tính toán khi sử dụng chú ý đặc trưng ngẫu nhiên, rẻ hơn nhiều so với softmax mặc định. (Nguồn ảnh: [Peng et al. 2021](https://arxiv.org/abs/2103.02143))._

**Chú ý nhân quả RFA** có token tại bước thời gian $t$ chỉ chú ý đến các khóa và giá trị trước đó $\{\mathbf{k}_i\}_{i \leq t}, \{\mathbf{v}_i\}_{i \leq t}$. Hãy sử dụng một bộ biến, $(\mathbf{S}_t \in \mathbb{R}^{2D \times d}, \mathbf{z} \in \mathbb{R}^{2D})$, để theo dõi lịch sử trạng thái ẩn tại bước thời gian $t$, tương tự như RNN:

$$
\begin{aligned}
&\text{causal-RFA}(\mathbf{q}_t, \{\mathbf{k}_i\}_{i \leq t}, \{\mathbf{v}_i\}_{i \leq t}) = \frac{\phi(\mathbf{q}_t)^\top \mathbf{S}_t}{\phi(\mathbf{q}_t) \cdot \mathbf{z}_t} \\
&\text{where }
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{k}_t)\otimes\mathbf{v}_t,
\quad
\mathbf{z}_t = \mathbf{z}_{t-1} + \phi(\mathbf{k}_t)
\end{aligned}
$$

trong đó $2D$ là kích thước của $\phi(.)$ và $D$ không được nhỏ hơn kích thước mô hình $d$ để có xấp xỉ hợp lý.

RFA dẫn đến việc tăng tốc đáng kể trong việc giải mã tự hồi quy và độ phức tạp bộ nhớ chủ yếu phụ thuộc vào việc lựa chọn $D$ khi xây dựng kernel $\phi(.)$.

Performer sửa đổi chú ý đặc trưng ngẫu nhiên bằng các bản đồ đặc trưng ngẫu nhiên dương để giảm lỗi ước tính. Nó cũng giữ cho các $\mathbf{w}_1, \dots, \mathbf{w}_D$ được lấy mẫu ngẫu nhiên trực giao để giảm thêm phương sai của bộ ước tính.

![So sánh lỗi xấp xỉ trong Performer](/posts/transformer-family-2/performer.png)
_So sánh lỗi xấp xỉ khi sử dụng (Trái) các đặc trưng i.i.d. so với các đặc trưng trực giao và (Phải) các đặc trưng ngẫu nhiên sin/cos so với các đặc trưng ngẫu nhiên dương. (Nguồn ảnh: [Choromanski et al. 2021](https://arxiv.org/abs/2009.14794))._

# Transformer cho Học tăng cường

Cơ chế tự chú ý tránh việc nén toàn bộ quá khứ vào một trạng thái ẩn có kích thước cố định và không bị ảnh hưởng nhiều bởi các gradient biến mất hoặc bùng nổ như RNN. Các nhiệm vụ Học tăng cường chắc chắn có thể hưởng lợi từ những đặc điểm này. _Tuy nhiên_, việc huấn luyện một Transformer ngay cả trong học có giám sát cũng khá khó khăn, chưa kể đến trong bối cảnh RL. Rốt cuộc, việc ổn định và huấn luyện một tác nhân LSTM tự nó đã có thể là một thách thức.

**Gated Transformer-XL** (**GTrXL**; [Parisotto, et al. 2019](https://arxiv.org/abs/1910.06764)) là một nỗ lực sử dụng Transformer cho RL. GTrXL đã thành công trong việc ổn định quá trình huấn luyện với hai thay đổi trên [Transformer-XL](#longer-attention-span-transformer-xl):

1.  Chuẩn hóa lớp chỉ được áp dụng cho luồng đầu vào trong một mô-đun dư, nhưng KHÔNG áp dụng cho luồng tắt. Một lợi ích chính của việc sắp xếp lại này là cho phép đầu vào ban đầu chảy từ lớp đầu tiên đến lớp cuối cùng.
2.  Kết nối dư được thay thế bằng một cơ chế _cổng (gating)_ kiểu GRU (Gated Recurrent Unit; [Chung et al., 2014](https://arxiv.org/abs/1412.3555)).

$$
\begin{aligned}
r &= \sigma(W_r^{(l)} y + U_r^{(l)} x) \\
z &= \sigma(W_z^{(l)} y + U_z^{(l)} x - b_g^{(l)}) \\
\hat{h} &= \tanh(W_g^{(l)} y + U_g^{(l)} (r \odot x)) \\
g^{(l)}(x, y) &= (1-z)\odot x + z\odot \hat{h}
\end{aligned}
$$

Các tham số của hàm cổng được khởi tạo rõ ràng để gần với một ánh xạ đồng nhất - đó là lý do tại sao có một thành phần $b_g$. Một $b_g > 0$ giúp tăng tốc độ học tập đáng kể.

![So sánh kiến trúc mô hình của Transformer-XL, Transformer-XL với chuẩn hóa lớp được sắp xếp lại và Gated Transformer-XL.](/posts/transformer-family-2/gated-transformer-XL.png)
_So sánh kiến trúc mô hình của Transformer-XL, Transformer-XL với chuẩn hóa lớp được sắp xếp lại và Gated Transformer-XL. (Nguồn ảnh: Hình 1 trong [Parisotto, et al. 2019](https://arxiv.org/abs/1910.06764))_

**Decision Transformer** (**DT**; [Chen et al 2021](https://arxiv.org/abs/2106.01345)) xây dựng các bài toán Học tăng cường như một quá trình _mô hình hóa chuỗi có điều kiện_, đưa ra các hành động tối ưu dựa trên lợi nhuận mong muốn, các trạng thái và hành động trong quá khứ. Do đó, việc sử dụng kiến trúc Transformer trở nên đơn giản. Decision Transformer dành cho [RL ngoài chính sách (off-policy RL)](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#key-concepts), trong đó mô hình chỉ có quyền truy cập vào một tập hợp cố định các quỹ đạo được thu thập bởi các chính sách khác.

Để khuyến khích mô hình học cách hành động để đạt được một lợi nhuận mong muốn, nó cung cấp cho mô hình lợi nhuận tương lai mong muốn $\hat{R} = \sum_{t'=t}^T r_{t'}$ thay vì phần thưởng hiện tại. Quỹ đạo bao gồm một danh sách các bộ ba, (lợi nhuận còn lại $\hat{R}_t, trạng thái $s_t$, hành động $a_t$), và nó được sử dụng như một chuỗi đầu vào cho Transformer:

$$
\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots, \hat{R}_T, s_T, a_T)
$$

Ba lớp tuyến tính được thêm vào và huấn luyện tương ứng cho lợi nhuận còn lại, trạng thái và hành động để trích xuất các nhúng token. Đầu dự đoán học cách dự đoán $a_t$ tương ứng với token đầu vào $s_t$. Quá trình huấn luyện sử dụng hàm mất mát entropy chéo cho các hành động rời rạc hoặc MSE cho các hành động liên tục. Việc dự đoán các trạng thái hoặc lợi nhuận còn lại không được phát hiện là giúp cải thiện hiệu suất trong các thí nghiệm của họ.

Các thí nghiệm đã so sánh DT với một số đường cơ sở thuật toán RL không có mô hình và cho thấy rằng:

- DT hiệu quả hơn so với sao chép hành vi trong chế độ dữ liệu thấp;
- DT có thể mô hình hóa rất tốt sự phân phối của lợi nhuận;
- Việc có một ngữ cảnh dài là rất quan trọng để có được kết quả tốt;
- DT có thể hoạt động với các phần thưởng thưa.

# Trích dẫn

Trích dẫn như sau:

> Weng, Lilian. (Tháng 1 năm 2023). Họ transformer phiên bản 2.0. Lil'Log. https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/.

Hoặc

```
@article{weng2023transformer,
  title   = "The Transformer Family Version 2.0",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io",
  year    = "2023",
  month   = "Jan",
  url     = "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/"
}
```

# Tài liệu tham khảo

1. Ashish Vaswani, et al. [“Attention is all you need.”](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) NIPS 2017.
2. Rami Al-Rfou, et al. [“Character-level language modeling with deeper self-attention.”](https://arxiv.org/abs/1808.04444) AAAI 2019.
3. Olah & Carter, [“Attention and Augmented Recurrent Neural Networks”](http://doi.org/10.23915/disti), Distill, 2016.
4. Sainbayar Sukhbaatar, et al. [“Adaptive Attention Span in Transformers”](https://arxiv.org/abs/1905.07799). ACL 2019.
5. Rewon Child, et al. [“Generating Long Sequences with Sparse Transformers”](https://arxiv.org/abs/1904.10509) arXiv:1904.10509 (2019).
6. Nikita Kitaev, et al. [“Reformer: The Efficient Transformer”](https://arxiv.org/abs/2001.04451) ICLR 2020.
7. Alex Graves. [“Adaptive Computation Time for Recurrent Neural Networks”](https://arxiv.org/abs/1603.08983)
8. Niki Parmar, et al. [“Image Transformer”](https://arxiv.org/abs/1802.05751) ICML 2018.
9. Zihang Dai, et al. [“Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context.”](https://arxiv.org/abs/1901.02860) ACL 2019.
10. Aidan N. Gomez, et al. [“The Reversible Residual Network: Backpropagation Without Storing Activations”](https://arxiv.org/abs/1707.04585) NIPS 2017.
11. Mostafa Dehghani, et al. [“Universal Transformers”](https://arxiv.org/abs/1807.03819) ICLR 2019.
12. Emilio Parisotto, et al. [“Stabilizing Transformers for Reinforcement Learning”](https://arxiv.org/abs/1910.06764) arXiv:1910.06764 (2019).
13. Rae et al. [“Compressive Transformers for Long-Range Sequence Modelling.”](https://arxiv.org/abs/1911.05507) 2019.
14. Press et al. [“Train Short, Test Long: Attention With Linear Biases Enables Input Length Extrapolation.”](https://arxiv.org/abs/2108.12409) ICLR 2022.
15. Wu, et al. [“DA-Transformer: Distance Aware Transformer”](https://aclanthology.org/2021.naacl-main.166) 2021.
16. Elabyad et al. [“Depth-Adaptive Transformer.”](https://arxiv.org/abs/1910.10073) ICLR 2020.
17. Schuster et al. [“Confident Adaptive Language Modeling”](https://arxiv.org/abs/2207.07061) 2022.
18. Qiu et al. [“Blockwise self-attention for long document understanding”](https://arxiv.org/abs/1911.02972) 2019
19. Roy et al. [“Efficient Content-Based Sparse Attention with Routing Transformers.”](https://arxiv.org/abs/2003.05997) 2021.
20. Ainslie et al. [“ETC: Encoding Long and Structured Inputs in Transformers.”](https://aclanthology.org/2020.emnlp-main.19/) EMNLP 2019.
21. Beltagy et al. [“Longformer: The long-document transformer.”](https://arxiv.org/abs/2004.05150) 2020.
22. Zaheer et al. [“Big Bird: Transformers for Longer Sequences.”](https://arxiv.org/abs/2007.14062) 2020.
23. Wang et al. [“Linformer: Self-Attention with Linear Complexity.”](https://arxiv.org/abs/2006.04768) arXiv preprint arXiv:2006.04768 (2020).
24. Tay et al. 2020 [“Sparse Sinkhorn Attention.”](https://arxiv.org/abs/2002.11296) ICML 2020.
25. Peng et al. [“Random Feature Attention.”](https://arxiv.org/abs/2103.02143) ICLR 2021.
26. Choromanski et al. [“Rethinking Attention with Performers.”](https://arxiv.org/abs/2009.14794) ICLR 2021.
27. Khandelwal et al. [“Generalization through memorization: Nearest neighbor language models.”](https://arxiv.org/abs/1911.00172) ICLR 2020.
28. Yogatama et al. [“Adaptive semiparametric language models.”](https://arxiv.org/abs/2102.02557) ACL 2021.
29. Wu et al. [“Memorizing Transformers.”](https://arxiv.org/abs/2203.08913) ICLR 2022.
30. Su et al. [“Roformer: Enhanced transformer with rotary position embedding.”](https://arxiv.org/abs/2104.09864) arXiv preprint arXiv:2104.09864 (2021).
31. Shaw et al. [“Self-attention with relative position representations.”](https://arxiv.org/abs/1803.02155) arXiv preprint arXiv:1803.02155 (2018).
32. Tay et al. [“Efficient Transformers: A Survey.”](https://arxiv.org/abs/2009.06732) ACM Computing Surveys 55.6 (2022): 1-28.
33. Chen et al., [“Decision Transformer: Reinforcement Learning via Sequence Modeling”](https://arxiv.org/abs/2106.01345) arXiv preprint arXiv:2106.01345 (2021).
