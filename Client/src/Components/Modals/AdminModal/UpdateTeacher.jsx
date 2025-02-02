/* eslint-disable react/no-unescaped-entities */
import PropTypes from "prop-types"; // Import PropTypes
import {
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Button,
} from "@nextui-org/react";
import { Link } from "react-router-dom";

const UpdateTeacher = ({ show, onHide, user }) => {
  return (
    <Modal
      isOpen={show}
      onClose={onHide}
      aria-labelledby="contained-modal-title-vcenter"
      isDismissable={false}
      isKeyboardDismissDisabled={true}
    >
      <ModalContent>
        <ModalHeader
          className="flex flex-col"
          id="contained-modal-title-vcenter"
        >
          Update Teacher Information
        </ModalHeader>
        <ModalBody>
          <p>Are you sure you want to update this teacher's information?</p>
        </ModalBody>
        <ModalFooter>
          <Button color="danger" variant="light" onClick={onHide}>
            Cancel
          </Button>
          <Link to={`/adminEditTeacher/${user?._id}`}>
            <Button color="primary">Update</Button>
          </Link>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

UpdateTeacher.propTypes = {
  show: PropTypes.bool.isRequired,
  onHide: PropTypes.func.isRequired,
  user: PropTypes.object,
};

export default UpdateTeacher;
