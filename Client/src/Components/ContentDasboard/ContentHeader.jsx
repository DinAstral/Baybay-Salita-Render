import React, { useContext, useEffect, useState } from "react";
import "./Content.css";
import { Tooltip } from "@nextui-org/react";
import { UserContext } from "../../../context/userContext";
import defaultProfilePic from "../../assets/profile.png";
import { Link } from "react-router-dom";

const ContentHeader = () => {
  const { user } = useContext(UserContext);
  const [profilePic, setProfilePic] = useState(defaultProfilePic);

  useEffect(() => {
    const updateProfilePic = () => {
      if (user && user.Picture) {
        const img = new Image();
        img.onload = () => setProfilePic(user.Picture);
        img.onerror = () => setProfilePic(defaultProfilePic);
        img.src = user.Picture;
      } else {
        setProfilePic(defaultProfilePic);
      }
    };

    updateProfilePic();
  }, [user]);

  return (
    <div className="w-[100%] px-9">
      <div className="flex gap-5 justify-end">
        <div className="flex items-center gap-1 justify-end">
          <Tooltip
            showArrow={true}
            content={
              <div className="px-1 py-2">
                <div className="text-sm font-bold">View Profile</div>
                <div className="text-xs">
                  This function will view your profile information.
                </div>
              </div>
            }
          >
            <Link to={`/${user?.role.toLowerCase()}Profile`}>
              <img
                src={profilePic}
                alt="Profile"
                className="w-16 h-16 rounded-full border border-gray-300"
              />
            </Link>
          </Tooltip>
          <div className="ml-4">
            {user ? (
              <>
                <h5 className="text-lg font-semibold">Hi {user.FirstName}!</h5>
                <h4 className="text-md text-gray-600">{user.role}</h4>
              </>
            ) : (
              <>
                <h5 className="text-lg font-semibold">Hi Guest!</h5>
                <h4 className="text-md text-gray-600">Role: N/A</h4>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContentHeader;
