def Train(model, args, train_dataloader, n_gpu):
    global_step = 0

    best_f1 = 0
    wait_step = 0
    model.train()
    global_step = 0
    stop_training = False

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in tqdm(enumerate(train_dataloader)):
            global_step += 1
            batch = [t.to(device) for t in batch]
            loss = model(batch, global_step)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()    # We have accumulated enought gradients
                model.zero_grad()
            if global_step % args.eval_period == 0:
                model.eval()
                f1 =  predict(args, model, eval_dataloader, eval_examples, eval_features, \
                                device, write_prediction=False)
                logger.info("%s: %.3f on epoch=%d" % (metric_name, f1*100.0, epoch))
                if best_f1 < f1:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    model = model.cuda()
                    best_f1 = f1
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if best_f1 > 0.1 and wait_step == args.wait_step:
                        stop_training = True
                model.train()
        if stop_training:
            break

    return model
