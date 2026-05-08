from training_scripts import train_sac

train_sac(
     num_episodes=800,
     batch_size=256,
     update_freq=1,
     model_save_freq=500
     )

