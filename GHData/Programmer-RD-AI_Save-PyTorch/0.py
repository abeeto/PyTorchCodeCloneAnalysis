def save(**kwargs: dict) -> None:
        """
        - save - it save the object with the {name}-{wandb-name}.pt and .pth
        ----------------------------------------------------
        - **kwargs - like Model().save(a="b")
        """
        torch.cuda.empty_cache()
        files_and_object = kwargs
        for files_and_object_key, files_and_object_val in tqdm(
            zip(files_and_object.keys(), files_and_object.values())
        ):  # iterate over the file and object
            torch.save(
                files_and_object_val, f"./models/{files_and_object_key}-{self.NAME}.pt"
            )  # Save the file in .pt
            torch.save(
                files_and_object_val, f"./models/{files_and_object_key}-{self.NAME}.pth"
            )  # Save the file in .pth
        torch.cuda.empty_cache()
