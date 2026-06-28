[INFO] got prompt
[INFO] Model SDXLClipModel prepared for dynamic VRAM loading. 1560MB Staged. 0 patches attached. Force pre-loaded 180 weights: 400 KB.
[INFO] Model SDXL prepared for dynamic VRAM loading. 4896MB Staged. 0 patches attached. Force pre-loaded 512 weights: 1197 KB.
  0%|                                                                                                                                                                                    | 0/60 [00:00<?, ?it/s,   Model Initializing ...  ]
[ERROR] !!! Exception during processing !!! Expected all tensors to be on the same device, but got mat2 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_mm)
[ERROR] Traceback (most recent call last):
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 542, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 341, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 315, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 303, in process_inputs
    result = f(**inputs)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\nodes.py", line 1584, in sample
    return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\nodes.py", line 1548, in common_ksampler
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\sample.py", line 74, in sample
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1444, in sample
    return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1334, in sample
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1316, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1254, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1229, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 999, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\k_diffusion\sampling.py", line 225, in sample_euler_ancestral
    denoised = model(x, sigmas[i] * s_in, **extra_args)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 639, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1202, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1209, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1212, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 619, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 210, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 218, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 334, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 191, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 235, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\modules\diffusionmodules\openaimodel.py", line 838, in forward
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, timesteps, context, y, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\modules\diffusionmodules\openaimodel.py", line 866, in _forward
    emb = self.time_embed(t_emb)
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\container.py", line 250, in forward
    input = module(input)
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1881, in _call_impl
    return inner()
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1842, in inner
    hook_result = hook(self, args, result)
  File "<frozen kqube_hook>", line 100, in post_hook
  File "<frozen kqube_hook>", line 109, in _cubic_transform
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "G:\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_mm)
