import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import os

# 设置赛博朋克风格
plt.style.use('dark_background')
colors = {
    'bg': '#0d0d0d',
    'entropy': '#ff0055', # 霓虹红
    'darwin': '#00ffaa',  # 霓虹绿
    'text': '#ffffff',
    'grid': '#333333'
}

def create_poster():
    # 1. 模拟数据 (这里为了快速生成，我们手动构造一组典型数据，或者读取真实数据)
    # 真实场景下应该调用模拟器，这里为了展示效果，构造一组完美的对比曲线
    steps = np.linspace(0, 5000, 500)
    
    # 递弱代偿宇宙 (Entropy Universe): P 指数衰减
    p_entropy = 0.98 ** (1 + steps/500)
    c_entropy = 1 + steps/200
    
    # 达尔文宇宙 (Darwin Universe): P 对数上升
    p_darwin = 0.98 + (0.019 * (1 - np.exp(-steps/1000)))
    c_darwin = 1 + steps/200

    # 2. 创建画布
    fig = plt.figure(figsize=(12, 16), facecolor=colors['bg'])
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1], hspace=0.3)

    # --- Header ---
    ax_header = fig.add_subplot(gs[0])
    ax_header.axis('off')
    ax_header.text(0.5, 0.7, "ENTROPY COMPENSATOR", 
                   ha='center', va='center', fontsize=40, color=colors['text'], fontweight='bold', fontname='Arial')
    ax_header.text(0.5, 0.4, "The Mathematics of Existence", 
                   ha='center', va='center', fontsize=18, color='#888888', fontstyle='italic')
    ax_header.text(0.5, 0.2, "v3.2 Multiverse Edition", 
                   ha='center', va='center', fontsize=12, color='#555555')

    # --- Main Chart ---
    ax_main = fig.add_subplot(gs[1])
    ax_main.set_facecolor(colors['bg'])
    
    # 绘制曲线
    ax_main.plot(steps, p_entropy, color=colors['entropy'], linewidth=3, label='Entropy Universe (Compensation)')
    ax_main.plot(steps, p_darwin, color=colors['darwin'], linewidth=3, linestyle='--', label='Darwin Universe (Evolution)')
    
    # 填充区域
    ax_main.fill_between(steps, p_entropy, 0, color=colors['entropy'], alpha=0.1)
    ax_main.fill_between(steps, p_darwin, 0, color=colors['darwin'], alpha=0.05)
    
    # 装饰
    ax_main.grid(True, color=colors['grid'], linestyle=':', linewidth=0.5)
    ax_main.set_ylabel('Degree of Existence (P)', fontsize=14, color='#aaaaaa')
    ax_main.set_xlabel('Evolution Steps (Time)', fontsize=14, color='#aaaaaa')
    
    # 标注关键点
    ax_main.annotate('Collapse Point', xy=(3000, p_entropy[300]), xytext=(3500, 0.6),
                     arrowprops=dict(facecolor='white', shrink=0.05),
                     color='white', fontsize=12)
    
    ax_main.legend(loc='upper right', frameon=False, fontsize=12)
    
    # 去除边框
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['bottom'].set_color('#555555')
    ax_main.spines['left'].set_color('#555555')

    # --- Footer (Philosophy) ---
    ax_footer = fig.add_subplot(gs[2])
    ax_footer.axis('off')
    
    quote = '"The degree of existence of all things declines over time;\ncomplexity is merely a compensatory measure to resist this decline."'
    ax_footer.text(0.5, 0.6, quote, 
                   ha='center', va='center', fontsize=16, color='#dddddd', style='italic', fontfamily='serif')
    
    ax_footer.text(0.5, 0.3, "— Wang Dongyue", 
                   ha='center', va='center', fontsize=14, color=colors['entropy'], fontweight='bold')

    # QR Code Placeholder (Optional)
    # rect = patches.Rectangle((0.45, 0.05), 0.1, 0.1, linewidth=1, edgecolor='white', facecolor='none')
    # ax_footer.add_patch(rect)
    # ax_footer.text(0.5, 0.1, "[ Scan to Run Simulation ]", ha='center', va='center', fontsize=10, color='#666666')

    # Save
    plt.savefig('social_media_poster.png', dpi=300, bbox_inches='tight', facecolor=colors['bg'])
    print("Poster generated: social_media_poster.png")

if __name__ == "__main__":
    try:
        create_poster()
    except Exception as e:
        print(f"Error: {e}")
