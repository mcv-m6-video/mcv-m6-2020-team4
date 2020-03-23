import numpy as np


class OpticalFlowBlockMatching:
    def __init__(self, type="FW", block_size=5, area_search=40, error_function="SSD", window_stride=5):
        """
        Class that performs optical flow with the block matching algorithm
        type :
            - "FW": for forward block matching
            - "BW": for bckward block matching
        block_size: size of the blocks
        area_search: number of pixels of expected movement in every direction
        error_function:
            - "SSD": sum of squared differences
            - "SAD": sum of absolute differences
        window_stride: step to look for matching block
        """
        if type == "FW" or type == "BW":
            self.type = type
        else:
            raise NameError("Unexpected type in OpticalFlowBlockMatching")
        self.block_size = block_size
        self.area_search = area_search
        self.window_stride = window_stride
        if error_function == "SSD":
            self.error_function = self.SSD
        elif error_function == "SAD":
            self.error_function = self.SAD
        else:
            raise NameError("Unexpected error function name in OpticalFlowBlockMatching")

    def SSD(self, block1, block2):
        return float(np.sum(np.power(block1 - block2, 2)))

    def SAD(self, block1, block2):
        return float(np.sum(np.abs(block1 - block2)))

    def compute_optical_flow(self, first_frame, second_frame):
        optical_flow = np.zeros((first_frame.shape[0], first_frame.shape[1], 3))
        if self.type == "FW":
            reference_image = first_frame.astype(float) / 255
            estimated_frame = second_frame.astype(float) / 255
        else:
            reference_image = second_frame.astype(float) / 255
            estimated_frame = first_frame.astype(float) / 255

        for i in range(self.block_size//2, reference_image.shape[0] - self.block_size // 2, self.block_size):
            for j in range(self.block_size//2, reference_image.shape[1] - self.block_size // 2, self.block_size):
                block_ref = reference_image[i - self.block_size // 2:i + self.block_size // 2 + 1, j - self.block_size // 2:j + self.block_size // 2 + 1, :]
                #optical_flow[i - self.block_size // 2:i + self.block_size // 2 + 1, j - self.block_size // 2:j + self.block_size // 2 + 1, :] = self.find_deviation_matching_block(block_ref, estimated_frame, (i,j))
                optical_flow[i, j, :] = self.find_deviation_matching_block(block_ref, estimated_frame, (i,j))
        return optical_flow

    def find_deviation_matching_block(self, block_ref, estimated_frame, position):
        min_likelihood = float('inf')
        min_direction = (0, 0)
        for i in range(max(self.block_size//2, position[0]-self.area_search), min(estimated_frame.shape[0] - self.block_size // 2, position[0]+self.area_search), self.window_stride):
            for j in range(max(self.block_size//2, position[1]-self.area_search), min(estimated_frame.shape[1] - self.block_size // 2, position[1]+self.area_search), self.window_stride):
                block_est = estimated_frame[i - self.block_size // 2:i + self.block_size // 2 + 1, j - self.block_size // 2:j + self.block_size // 2 + 1, :]
                likelihood = self.error_function(block_ref, block_est)
                if likelihood < min_likelihood:
                    min_likelihood = likelihood
                    min_direction = (i,j) # TODO: SURE?
                elif likelihood == min_likelihood and np.sum(np.power(min_direction, 2)) > j ** 2 + i ** 2:
                    min_direction = (i,j) # TODO: SURE?
        ret_block = np.ones((self.block_size, self.block_size, 3))
        ret_block[:, :, 0] = min_direction[0] # TODO: SURE?
        ret_block[:, :, 1] = min_direction[1] # TODO: SURE?
        #return ret_block
        return [min_direction[0], min_direction[1], 1]
