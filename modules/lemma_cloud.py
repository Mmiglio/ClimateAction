# Dependencies
import warnings
import os
import numpy as np
from random import Random
from operator import itemgetter
import matplotlib
import matplotlib.colors as mcolors

from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont

from wordcloud.query_integral_image import query_integral_image
from wordcloud import WordCloud

# Define filesystem paths
FILE = os.path.dirname(__file__)
FONT_PATH = os.environ.get('FONT_PATH', os.path.join(FILE, '../resources/DroidSansMono.ttf'))

# Get default colors
colors = [*mcolors.TABLEAU_COLORS.values()]


# Class for computing word positioning
class IntegralOccupancyMap(object):
    def __init__(self, height, width, mask):
        self.height = height
        self.width = width
        if mask is not None:
            self.integral = np.cumsum(np.cumsum(255 * mask, axis=1), axis=0).astype(np.uint32)
        else:
            self.integral = np.zeros((height, width), dtype=np.uint32)

    def sample_position(self, size_x, size_y, random_state):
        return query_integral_image(self.integral, size_x, size_y, random_state)

    def update(self, img_array, pos_x, pos_y):
        partial_integral = np.cumsum(np.cumsum(img_array[pos_x:, pos_y:], axis=1), axis=0)
        # paste recomputed part into old image
        # if x or y is zero it is a bit annoying
        if pos_x > 0:
            if pos_y > 0:
                partial_integral += (self.integral[pos_x - 1, pos_y:] - self.integral[pos_x - 1, pos_y - 1])
            else:
                partial_integral += self.integral[pos_x - 1, pos_y:]
        if pos_y > 0:
            partial_integral += self.integral[pos_x:, pos_y - 1][:, np.newaxis]

        self.integral[pos_x:, pos_y:] = partial_integral
        
# Define function for colouring word according to pos tag
def color_pos_tag(pos_tag):
    return {'N': colors[0], 'V': colors[1], 'A': colors[2], 'R': colors[3]}.get(pos_tag)
        
# Define wordcloud of lemmas class
class LemmaCloud(WordCloud):
    
    # Overwrite constructor
    def __init__(self, *args, **kwargs):
        # Call parent constructor
        super().__init__(font_path = FONT_PATH, *args, **kwargs)

    # Overwrite generate from frequencies method
    def generate_from_frequencies(self, scores_in, max_font_size=None):
        """Create a word_cloud from words and frequencies.

        Parameters
        ----------
        scores_in : dict from tuple to float
            A contains lemmas and associated score.

        max_font_size : int
            Use this font-size instead of self.max_font_size

        Returns
        -------
        self

        """
        
        # Check length of scores
        if len(scores_in) <= 0:
            raise ValueError("We need at least 1 word to plot a word cloud, "
                             "got %d." % len(scores_in))
        
        # Make sure scores are sorted and normalized
        scores_in = sorted(scores_in.items(), key=itemgetter(1), reverse=True)
        scores_in = scores_in[:self.max_words]  # Slice down to max words

        # Largest entry will be the 1st
        max_score = float(scores_in[0][1])

        # Normalize
        scores_norm = [((word, tag), score / max_score) for (word, tag), score in scores_in]

        # Set random state
        if self.random_state is not None:
            random_state = self.random_state
        else:
            random_state = Random()

        # Set boolean mask
        if self.mask is not None:
            boolean_mask = self._get_bolean_mask(self.mask)
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            boolean_mask = None
            height, width = self.height, self.width
        occupancy = IntegralOccupancyMap(height, width, boolean_mask)

        # Create image
        img_grey = Image.new("L", (width, height))
        draw = ImageDraw.Draw(img_grey)
        img_array = np.asarray(img_grey)
        font_sizes, positions, orientations, colors = [], [], [], []

        last_score = 1.

        # If not provided use default font_size
        if max_font_size is None:
            max_font_size = self.max_font_size

        # Figure out a good font size by trying to draw with
        if max_font_size is None:
            
            # Just the first two words
            if len(scores_norm) == 1:
                # We only have one word. We make it big!
                font_size = self.height
            else:
                # Recursive call: this sets layout_
                self.generate_from_frequencies(dict(scores_norm[:2]), max_font_size=self.height)
                # Find font sizes
                sizes = [x[1] for x in self.layout_]
                try:
                    font_size = int(2 * sizes[0] * sizes[1] / (sizes[0] + sizes[1]))
                # Quick fix for if self.layout_ contains less than 2 values
                # On very small images it can be empty
                except IndexError:
                    try:
                        font_size = sizes[0]
                    except IndexError:
                        raise ValueError(
                            "Couldn't find space to draw. Either the Canvas size"
                            " is too small or too much of the image is masked "
                            "out.")
        # Case font size has been manually set
        else:
            font_size = max_font_size

        # Set self.words_ here, because we called generate_from_frequencies
        # above... hurray for good design?
        self.words_ = {word: score for (word, tag), score in scores_norm}

        # Check repetitiorn
        if self.repeat and len(scores_norm) < self.max_words:
            # Pad frequencies with repeating words.
            times_extend = int(np.ceil(self.max_words / len(scores_norm))) - 1
            # Get smallest frequency
            scores_org = list(scores_norm)
            downweight = scores_norm[-1][1]
            for i in range(times_extend):
                scores_norm.extend([((word, tag), freq * downweight ** (i + 1))
                                    for (word, tag), score in scores_org])

        # start drawing grey image
        for (word, tag), score in scores_norm:
            # Do not show 0 score lemmas
            if score == 0:
                continue
            # Select the font size
            rs = self.relative_scaling
            if rs != 0:
                font_size = int(round((rs * (score / float(last_score))
                                       + (1 - rs)) * font_size))
            if random_state.random() < self.prefer_horizontal:
                orientation = None
            else:
                orientation = Image.ROTATE_90
            tried_other_orientation = False
            while True:
                # Try to find a position
                font = ImageFont.truetype(self.font_path, font_size)
                # Transpose font optionally
                transposed_font = ImageFont.TransposedFont(font, orientation=orientation)
                # Get size of resulting text
                box_size = draw.textsize(word, font=transposed_font)
                # Find possible places using integral image:
                result = occupancy.sample_position(box_size[1] + self.margin, box_size[0] + self.margin, random_state)
                # Either we found a place or font-size went too small
                if result is not None or font_size < self.min_font_size:
                    break
                # If we didn't find a place, make font smaller, but first try to rotate!
                if not tried_other_orientation and self.prefer_horizontal < 1:
                    orientation = (Image.ROTATE_90 if orientation is None else
                                   Image.ROTATE_90)
                    tried_other_orientation = True
                # Make font smaller
                else:
                    font_size -= self.font_step
                    orientation = None
            
            # Case we were unable to draw any more
            if font_size < self.min_font_size:
                break

            # Define position of the text
            x, y = np.array(result) + self.margin // 2
            # Actually draw the text
            draw.text((y, x), word, fill="white", font=transposed_font)
            positions.append((x, y))
            orientations.append(orientation)
            font_sizes.append(font_size)
            # Color according to POS tag
            colors.append(color_pos_tag(tag))
            # colors.append(self.color_func(word, font_size=font_size,
            #                               position=(x, y),
            #                               orientation=orientation,
            #                               random_state=random_state,
            #                               font_path=self.font_path))
            # recompute integral image
            if self.mask is None:
                img_array = np.asarray(img_grey)
            else:
                img_array = np.asarray(img_grey) + boolean_mask
            # Recompute bottom right
            # The order of the cumsum's is important for speed ?!
            occupancy.update(img_array, x, y)
            last_score = score
        # Set layout
        self.layout_ = list(zip([(word, score) for (word, tag), score in scores_norm], 
                                font_sizes, positions,orientations, colors))
        # Return object itself
        return self