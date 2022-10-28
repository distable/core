import os
import sys
import traceback

import torch

from PIL import Image

import src_plugins.sd1111_plugin.options
import src_plugins.sd1111_plugin.sd_paths
import src_plugins.sd1111_plugin.SDState
from src_plugins.sd1111_plugin import devices
import src_plugins.sd1111_plugin.sd_textinv_dataset
from src_plugins.sd1111_plugin.image_embedding import (embedding_from_b64,
                                                       extract_image_data_embed)

import src_core.paths


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None

    def save(self, filename):
        embedding_data = {
            "string_to_token"   : {"*": 265},
            "string_to_param"   : {"*": self.vec},
            "name"              : self.name,
            "step"              : self.step,
            "sd_checkpoint"     : self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class EmbeddingDatabase:
    def __init__(self, embeddings_dir):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.dir_mtime = None
        self.embeddings_dir = embeddings_dir

    def register_embedding(self, embedding, model):
        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenizer([embedding.name], add_special_tokens=False)['input_ids'][0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def load_textual_inversion_embeddings(self):
        from src_plugins.sd1111_plugin.SDPlugin import SDPlugin
        mt = os.path.getmtime(self.embeddings_dir)
        if self.dir_mtime is not None and mt <= self.dir_mtime:
            return

        self.dir_mtime = mt
        self.ids_lookup.clear()
        self.word_embeddings.clear()

        def process_file(path, filename):
            name = os.path.splitext(filename)[0]

            data = []

            if os.path.splitext(filename.upper())[-1] in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
                embed_image = Image.open(path)
                if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                    data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                    name = data.get('name', name)
                else:
                    data = extract_image_data_embed(embed_image)
                    name = data.get('name', name)
            else:
                data = torch.load(path, map_location="cpu")

            # textual inversion embeddings
            if 'string_to_param' in data:
                param_dict = data['string_to_param']
                if hasattr(param_dict, '_parameters'):
                    param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
                assert len(param_dict) == 1, 'embedding file has multiple terms in it'
                emb = next(iter(param_dict.items()))[1]
            # diffuser concepts
            elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
                assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

                emb = next(iter(data.values()))
                if len(emb.shape) == 1:
                    emb = emb.unsqueeze(0)
            else:
                raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

            vec = emb.detach().to(devices.device, dtype=torch.float32)
            embedding = Embedding(vec, name)
            embedding.step = data.get('step', None)
            embedding.sd_checkpoint = data.get('hash', None)
            embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
            self.register_embedding(embedding, SDPlugin.sdmodel)

        for fn in os.listdir(self.embeddings_dir):
            try:
                fullfn = os.path.join(self.embeddings_dir, fn)

                if os.stat(fullfn).st_size == 0:
                    continue

                process_file(fullfn, fn)
            except Exception:
                print(f"Error loading emedding {fn}:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                continue

        print(f"Loaded a total of {len(self.word_embeddings)} textual inversion embeddings.")
        print("Embeddings:", ', '.join(self.word_embeddings.keys()))

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None


def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
    from src_plugins.sd1111_plugin import SDPlugin

    cond_model = src_plugins.sd1111_plugin.SDState.sdmodel.cond_stage_model
    embedding_layer = cond_model.wrapped.transformer.text_model.embeddings

    ids = cond_model.tokenizer(init_text, max_length=num_vectors_per_token, return_tensors="pt", add_special_tokens=False)["input_ids"]
    embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=devices.device)

    for i in range(num_vectors_per_token):
        vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    fn = os.path.join(src_core.paths.plug_embeddings, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = Embedding(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn


