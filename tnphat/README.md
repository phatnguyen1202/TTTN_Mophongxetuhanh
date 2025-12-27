# 🚗 Xe Tự Hành Phát Hiện Chướng Ngại Vật với Q-Learning

Project mô phỏng xe tự hành trong môi trường 2D với chướng ngại vật, sử dụng **thuật toán Q-Learning** để học cách di chuyển từ điểm xuất phát đến đích.

![Demo](demo_2_full.png)

## 📋 Mô tả

Xe tự hành học cách tìm đường đi tối ưu trong lưới 10x10 với các chướng ngại vật ngẫu nhiên. Agent sử dụng Q-Learning với reward shaping để:
- ✅ Di chuyển từ góc trên trái (0,0) đến góc dưới phải (9,9)
- ✅ Tránh va chạm chướng ngại vật
- ✅ Tối ưu hóa số bước di chuyển

## 🎯 Kết quả

- **Tỷ lệ thành công**: 52-60% (sau 1000 episodes)
- **Số bước trung bình**: ~18-22 bước
- **Reward trung bình**: ~43-117

## 📁 Cấu trúc Project

```
autonomous_car_qlearning/
│
├── environment_v2.py        # Môi trường lưới 2D (cải tiến)
├── qlearning_agent.py       # Agent Q-Learning
├── visualizer.py            # Visualization với Pygame
├── train_improved.py        # Script training cải tiến
├── demo_multiple.py         # Chạy nhiều demo và chọn tốt nhất
├── demo_visualization.py    # Tạo hình ảnh visualization
│
├── trained_model.pkl        # Model đã train
├── demo_2_full.png          # Hình ảnh demo full path
├── demo_2_steps.png         # Hình ảnh demo từng bước
│
├── requirements.txt         # Dependencies
└── README.md               # File này
```

## 🚀 Cài đặt và Chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc:

```bash
pip install numpy pygame matplotlib
```

### 2. Train agent mới

```bash
python train_improved.py
```

**Cấu hình training:**
- Episodes: 1000
- Learning rate: 0.2
- Discount factor: 0.99
- Epsilon decay: 0.998
- Grid size: 10x10

### 3. Tạo demo visualization

```bash
python demo_multiple.py
```

Script này sẽ:
- Chạy 10 demo với agent đã train
- Chọn demo thành công tốt nhất
- Tạo 2 hình ảnh visualization

## 🧠 Thuật toán Q-Learning

### Công thức cập nhật Q-value

```
Q(s,a) ← Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
```

**Trong đó:**
- `s`: State hiện tại (vị trí xe)
- `a`: Action (lên/xuống/trái/phải)
- `r`: Reward nhận được
- `s'`: State tiếp theo
- `α`: Learning rate = 0.2
- `γ`: Discount factor = 0.99

### Epsilon-Greedy Policy

- **Exploration** (ε): Chọn action ngẫu nhiên
- **Exploitation** (1-ε): Chọn action tốt nhất từ Q-table
- Epsilon giảm dần từ 1.0 → 0.05

## 🎁 Reward System

| Sự kiện | Reward |
|---------|--------|
| Đến đích thành công | +100 |
| Di chuyển gần đích hơn | +1 |
| Di chuyển xa đích hơn | -1 |
| Va chướng ngại vật/ra ngoài | -10 |

## 📊 Visualization

Project cung cấp 2 loại visualization:

### 1. Pygame Interactive (visualizer.py)
- Hiển thị real-time
- Có thể tương tác (ESC, SPACE)
- Cần display server

### 2. Static Images (demo_visualization.py)
- Tạo hình ảnh PNG
- Hiển thị từng bước
- Hiển thị full path
- Không cần display

## 🔧 Cải tiến so với version ban đầu

1. **Reward Shaping**: Thêm reward dựa trên khoảng cách Manhattan
2. **Giảm chướng ngại vật**: Từ 10-15 xuống 5-8
3. **Tăng learning rate**: 0.1 → 0.2
4. **Tăng discount factor**: 0.95 → 0.99
5. **Epsilon decay chậm hơn**: 0.995 → 0.998
6. **Không chặn đường chéo chính**: Luôn có đường đi khả thi

## 📈 Kết quả Training

```
Episode 100/1000
  ✓ Tỷ lệ thành công: 10.0%
  📊 Reward trung bình: 5.84
  
Episode 500/1000
  ✓ Tỷ lệ thành công: 49.0%
  📊 Reward trung bình: 56.12
  
Episode 1000/1000
  ✓ Tỷ lệ thành công: 52.0%
  📊 Reward trung bình: 43.49
```

## 🎨 Màu sắc trong Visualization

- 🟦 **Xanh dương**: Xe (Agent)
- 🟨 **Vàng**: Đích (Goal)
- ⬛ **Đen**: Chướng ngại vật
- ⬜ **Trắng**: Ô trống
- 🟩 **Xanh lá**: Xe đến đích thành công
- 🔴 **Đường đỏ**: Đường đi của xe

## 💡 Cải tiến tiếp theo

1. **Deep Q-Network (DQN)**: Thay Q-table bằng neural network
2. **Double DQN**: Giảm overestimation của Q-values
3. **Prioritized Experience Replay**: Học từ experience quan trọng
4. **Multi-agent**: Nhiều xe cùng lúc
5. **Dynamic obstacles**: Chướng ngại vật di chuyển
6. **Continuous space**: Không gian liên tục
7. **Vision sensors**: Xe có camera/lidar

## 📚 Tham khảo

- **Q-Learning**: Watkins, C. J., & Dayan, P. (1992)
- **Reinforcement Learning**: Sutton & Barto (2018)
- **Reward Shaping**: Ng, A. Y., Harada, D., & Russell, S. (1999)

## 👨‍💻 Code Structure

### Environment (environment_v2.py)
```python
- State: Vị trí xe (0-99 trong lưới 10x10)
- Actions: 4 hướng (lên, xuống, trái, phải)
- Reward: +100 (đích), +1/-1 (di chuyển), -10 (va chạm)
```

### Agent (qlearning_agent.py)
```python
- Q-table: 100 states × 4 actions
- Policy: Epsilon-greedy
- Update: Q-learning formula
```

### Visualizer (visualizer.py)
```python
- Grid rendering với Pygame
- Real-time animation
- Info panel với metrics
```

## 📝 License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu

## 🙏 Acknowledgments

Project này được tạo ra cho mục đích học tập về Reinforcement Learning và Q-Learning.

---

**Được tạo bởi**: AI Assistant  
**Ngày tạo**: 2025  
**Công nghệ**: Python, Q-Learning, Pygame, Matplotlib
