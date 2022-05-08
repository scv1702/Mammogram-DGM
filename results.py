import matplotlib.pylab as plt
from train import loss_G_list, loss_A_list, loss_R1_list, loss_R2_list, loss_R3_list, loss_TV_list

plt.title(f"Total loss")
plt.plot(loss_G_list, label='loss G')
plt.plot(loss_A_list, label='loss D')
plt.legend()
plt.show()

plt.title(f"Reconstruction loss")
plt.plot(loss_R1_list, label='loss R1')
plt.plot(loss_R2_list, label='loss R2')
plt.plot(loss_R3_list, label='loss R3')
plt.legend()
plt.show()

plt.title(f"Total Variation loss")
plt.plot(loss_TV_list, label='loss TV')
plt.legend()
plt.show()

plt.title(f"Adversarial loss")
plt.plot(loss_A_list, label='loss A')
plt.legend()
plt.show()