"""
Generate Jensen's Inequality visualization for logarithm function.
This plot demonstrates the key mathematical concept behind the ELBO derivation in VAEs.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_jensen_plot():
    """Generate and save Jensen's inequality plot for log function."""
    
    # Set up the figure with a clean, academic style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Define the log function
    x = np.linspace(0.1, 4, 1000)
    log_x = np.log(x)

    # Plot the log curve
    ax.plot(x, log_x, 'b-', linewidth=3, label=r'$f(x) = \log(x)$')

    # Choose specific points to demonstrate Jensen's inequality
    x1, x2 = 0.5, 3.0
    y1, y2 = np.log(x1), np.log(x2)

    # Plot the points
    ax.plot([x1, x2], [y1, y2], 'ro', markersize=10, label='Sample points')

    # Draw the secant line between the two points
    secant_x = np.linspace(x1, x2, 100)
    slope = (y2 - y1) / (x2 - x1)
    secant_y = y1 + slope * (secant_x - x1)
    ax.plot(secant_x, secant_y, 'r--', linewidth=2, label='Secant line')

    # Choose a convex combination point
    lambda_val = 0.6  # weight for x1
    x_conv = lambda_val * x1 + (1 - lambda_val) * x2
    y_conv_actual = np.log(x_conv)  # f(λx₁ + (1-λ)x₂)
    y_conv_linear = lambda_val * y1 + (1 - lambda_val) * y2  # λf(x₁) + (1-λ)f(x₂)

    # Plot the convex combination point
    ax.plot(x_conv, y_conv_actual, 'go', markersize=12, label=r'$f(\lambda x_1 + (1-\lambda) x_2)$')
    ax.plot(x_conv, y_conv_linear, 'mo', markersize=12, label=r'$\lambda f(x_1) + (1-\lambda) f(x_2)$')

    # Draw vertical line to show the inequality
    ax.plot([x_conv, x_conv], [y_conv_actual, y_conv_linear], 'k-', linewidth=3, alpha=0.8)

    # Add arrow to show the inequality direction (moved down slightly)
    ax.annotate('', xy=(x_conv - 0.05, y_conv_linear - 0.02), xytext=(x_conv - 0.05, y_conv_actual - 0.02),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))

    # Add annotations for the specific points (repositioned to avoid overlaps)
    ax.annotate(f'$x_1 = {x1}$', xy=(x1, y1), xytext=(x1+0.1, y1-0.6),
                fontsize=12, ha='center', 
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    ax.annotate(f'$x_2 = {x2}$', xy=(x2, y2), xytext=(x2-0.2, y2+0.3),
                fontsize=12, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.6))
    ax.annotate(f'$\\lambda x_1 + (1-\\lambda)x_2 = {x_conv:.1f}$', xy=(x_conv, y_conv_actual), 
                xytext=(x_conv-0.8, y_conv_actual-0.1),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.6))

    # Add Jensen's inequality explanation (repositioned to avoid overlap)
    ax.text(2.8, -0.8, 
            "Jensen's Inequality:\n" + 
            r'$f(\lambda x_1 + (1-\lambda) x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)$' + '\n' + 
            r'(For concave functions like $\log$, inequality is $\geq$)',
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8))

    # Add mathematical foundation text (moved further down to avoid overlaps)
    ax.text(2.8, 0.2, r'For concave functions like $\log(x)$:', fontsize=13, weight='bold', ha='center')
    ax.text(2.8, -0.1, r'$\log(\lambda x_1 + (1-\lambda) x_2) \geq \lambda \log(x_1) + (1-\lambda) \log(x_2)$', 
            fontsize=11, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    # Set labels and title
    ax.set_xlabel('x', fontsize=14, weight='bold')
    ax.set_ylabel('f(x) = log(x)', fontsize=14, weight='bold')
    ax.set_title('Jensen\'s Inequality for the Logarithm Function\n(Foundation of ELBO Derivation in Variational Inference)', 
                 fontsize=16, weight='bold', pad=25)

    # Set axis limits with more space
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-1.6, 1.6)

    # Add legend (repositioned to avoid covering content)
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.85))

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Add VAE connection explanation (repositioned to top-left)
    explanation = (
        "Application in Variational Autoencoders:\n"
        "• Log function is concave (curves downward)\n"
        "• Jensen's inequality enables ELBO derivation\n"
        "• Key step: $\\log \\mathbb{E}[X] \\geq \\mathbb{E}[\\log X]$\n"
        "• This gives us: $\\log p(x) \\geq \\text{ELBO}$"
    )

    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", 
            facecolor="lightgreen", alpha=0.9))

    plt.tight_layout()
    
    # Save with relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    output_path = os.path.join(project_root, 'src', 'assets', 'images', 'jensen_inequality_log.png')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Jensen's inequality plot saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_jensen_plot()