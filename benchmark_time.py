
def benchmark(model, class_names=class_names, image_size=IMG_SIZE):
  file_count = 0
  infer_times = []
  init_timer = 0

  for image in val_data:
    if file_count < 1 :
        init_timer_start = time.time()
        pred = model.predict(image[0])
        pred_class = class_names[int(np.argmax(pred[0]))]
        init_timer_end = time.time()
        init_timer = init_timer_end - init_timer_start
        file_count+=1
    else:
        timer_start = time.time()
        pred = model.predict(image[0])
        pred_class = class_names[int(np.argmax(pred[0]))]
        timer_end = time.time()
        infer_times.append((timer_end - timer_start))
        file_count+=1

  return init_timer, np.mean(infer_times), np.std(infer_times)